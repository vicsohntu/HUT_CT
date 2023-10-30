import glob
from itertools import chain
import os,sys
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from core.dataloader import *
from core.utils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
from core.utils import test_single_volume
from core.criterions import FocalLoss, cross_entropy_dice, hard_cross_entropy, dice, dice_score, dice_loss_wgt, weighted_dice_loss
from core.dataloader import *
from core.eval_utils import *
from core.eval import *
from vit_pytorch.hut import CVITUNETUP2DCHUNK
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname is not None:
            if classname.find("Conv3d") != -1:
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif classname.find("ConvTranspose3d") != -1:
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif classname.find("Linear") != -1:
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif classname.find("BatchNorm3d") != -1:
                torch.nn.init.kaiming_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def labeltensor(alist):
    labelist=list()
    for aa in alist:
        labelist.append(int(aa.split("XPTM")[1]))
    return torch.tensor(labelist)

def znormalize(data, mean, std):
    norm_data = (data - mean) / std
    return norm_data.float()

lr = 3e-4 
weight_decay=1e-7
gamma = 0.7
seed = 12345
seed_everything(seed)
epochs=1001
num_classes=2
channel=1
patch_size=4
image_size=(256,256)
batchsize=1
epochstart=0
linux=True
autostop=124
cnt=0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
ATLAS_MEAN= 30.20063; ATLAS_SD= 35.221165
model =CVITUNETUP2DCHUNK(
    in_channels=4, out_channels=2, dropout=0.15, tr_dropout=0.25 
    )

abc=torch.randn(1,4, 256,256)
t,aa,bb=model(abc)
print(model)
decay_epoch=epochs//4
model = model.to(device)

optimgen = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.5, 0.99), weight_decay=weight_decay)
criterion_sub = nn.CrossEntropyLoss(weight=torch.tensor([0.15, 0.85])).to(device)
criterion_foc = FocalLoss(gamma=2, reduction='mean').to(device)
dice_loss = DiceLoss(num_classes)
current_dice=0.

steps=int(epochs/10)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimgen, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)
lambdaseg=0.6
lambdace=0.4
lambdacls=1e-6 
lambdacls_low=1e-8 
clsstep=(lambdacls-lambdacls_low)/decay_epoch
decay=True
print(clsstep)
trg_path="./data/isles2018_trg0"
trgset = tio.SubjectsDataset(make_sublist_isles2018(trg_path), transform = transformisles(1))
if linux:
    loader = DataLoader(trgset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=4)
else:
    loader = DataLoader(trgset, batch_size=batchsize, shuffle=True, drop_last=True)
tst_path="./data/isles2018_tst0"
tstset = tio.SubjectsDataset(make_sublist_isles2018(tst_path), transform = transformisles(0))

tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)

samplesize=int(len(trgset)/batchsize)
print("Dataset size: ", samplesize)
if epochstart==0:
    model.apply(weights_init_normal)
    print("Model weights randomised normally")
dsctloss=0.0
cetloss=0.0
dsctlossp=0.0
cetlossp=0.0
dsctlossd=0.0
cetlossd=0.0
dscweight=0.0
ceweight=1.
for epoch in range(epochs):
    model.train()
    starttime=time.time()
    totalloss=0.
    batchrun=0.
    for i_batch, sample in enumerate(loader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        for b in range(depth):
            stack_vol_=stack_vol[:,:,:,:,b].to(device)
            gseg_=gseg[:,:,:,:,b].to(device)
            optimgen.zero_grad()
            oseg, sm, lg = model(stack_vol_)
            celoss = criterion_sub(oseg, gseg_[0])
            dscloss= dice_loss(oseg, gseg_[0])
            sm=torch.softmax(sm, dim=-1)
            lg=torch.softmax(lg,dim=-1)
            clsloss=-(sm*torch.log(lg)).sum()
            lossgen=ceweight*celoss +dscweight*dscloss + lambdacls*clsloss ### best performance
            lossgen.backward()
            optimgen.step()
            totalloss+=lossgen.item()
            dsctloss+=dscloss.item()
            cetloss+=celoss.item()            
        batchrun=batchrun+(i_batch+1)*depth
        print("Iter: ", i_batch, "Current ave loss: ", totalloss/(batchrun),  "cls loss: ", clsloss.item())
    elapsedtime=time.time()-starttime
    dsctlossd=dsctloss-dsctlossp
    dsctlossp=dsctloss
    cetlossd=cetloss-cetlossp
    cetlossp=cetloss
    if dsctlossd>0.0 and cetlossd>0.0:
        dscweight=dsctlossd/(dsctlossd+cetlossd)
        ceweight=cetlossd/(dsctlossd+cetlossd)
    print("Epoch: ", epoch+epochstart, " Loss Average: " , totalloss/(batchrun), " DSC delta ", dsctlossd, " CE delta ", cetlossd,  " DSC weight ", dscweight, " CE weight ", ceweight, " Time Taken: ", elapsedtime) 
    if lambdacls> lambdacls_low+clsstep and decay:
        lambdacls=lambdacls-clsstep
    if lambdacls< 0.:
        lambdacls=lambdacls_low
    print(lambdacls)
    if epoch%5==0:
        with torch.no_grad():
            mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall=metric_eval_cvitcls_isles2018slice(model, tstloader, norm=False, device=device, num_classes=num_classes)
        original_stdout =sys.stdout
        print('Validation : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %f mean_recall : %f' % (mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
        if current_dice<mean_dice:
            cnt=0
            current_dice=mean_dice
            torch.save(model.state_dict(), 'results/CVITUNETUP2DmixCLS_model9b.pt')
            f=open("results/CVITUNETUP2DmixCLS_model9b.txt", "a") 
            f.write('Validation epoch %d : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %f mean_recall : %f \n' % (epoch, mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
            f.close()
        elif cnt<autostop:
            cnt=cnt+1
        else:
            print("Exceed autostop limit...")
            sys.exit()

