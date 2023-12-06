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
from core.criterions import cross_entropy_dice, hard_cross_entropy, dice, dice_score, FocalLoss, dice_loss_wgt, weighted_dice_loss
from core.dataloader import *
from core.eval_utils import *
from core.eval import *
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
from torch.nn import init
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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

lr = 4e-4 #used in paper
weight_decay=1e-5 #used in paper
gamma = 0.7
seed = 12345
seed_everything(seed)
epochs=501
num_classes=2
channel=4
patch_size=4
image_size=(128,128,128)
batchsize=1
epochstart=0
linux=True
autostop=124
cnt=0
current_dice=0.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import sys

_, model = TransBTS(dataset='isles', _conv_repr=True, _pe_type="learned")


print(model)
decay_epoch=epochs//2
model = model.to(device)
optimgen = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.5, 0.99), weight_decay=weight_decay)
criterion_sub = nn.CrossEntropyLoss().to(device)
dice_loss = DiceLoss(num_classes)
steps=int(epochs/10)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimgen, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)

lambdaseg=0.1
lambdace=0.9
trg_path="./data/processed128x128x128/isles2018_trg9"
trgset = tio.SubjectsDataset(make_sublist_isles2018_2(trg_path), transform = transformisles(1))
if linux:
    loader = DataLoader(trgset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=1)
else:
    loader = DataLoader(trgset, batch_size=batchsize, shuffle=True, drop_last=True) #windows
tst_path="./data/processed128x128x128/isles2018_tst9"
tstset = tio.SubjectsDataset(make_sublist_isles2018_2(tst_path), transform = transformisles(0))

tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)
samplesize=int(len(trgset)/batchsize)
print("Dataset size: ", samplesize)
if epochstart==0:
    init_weights(model, init_type='kaiming', gain=0.02)
    print("Model weights randomised normally")

model.train()
for epoch in range(epochs):
    starttime=time.time()
    totalloss=0.0
    for i_batch, sample in enumerate(loader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        gseg = sample['segvol']['data'].long()
        stack_vol_=stack_vol.to(device)
        gseg_=gseg.to(device)
        optimgen.zero_grad()
        oseg = model(stack_vol_)
        celoss = criterion_sub(oseg, gseg_[0])
        dscloss= dice_loss(oseg, gseg_[0])
        lossgen=celoss +dscloss
        lossgen.backward()
        optimgen.step()
        totalloss+=lossgen.item()
        print("Iter: ", i_batch, "Current ave loss: ", totalloss/(i_batch+1),  "ce loss: ", celoss.item())

    elapsedtime=time.time()-starttime
    print("Epoch: ", epoch, " Loss Average: " , totalloss/(i_batch+1), " Time Taken: ", elapsedtime) 
    if epoch%2==0:
        with torch.no_grad():
            mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall=metric_eval_3dunet_isles2018(model, tstloader, norm=False, device=device, num_classes=num_classes)
        original_stdout =sys.stdout
        print('Validation : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %f mean_recall : %f' % (mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
        if current_dice<mean_dice:
            cnt=0
            current_dice=mean_dice
            f2rem=glob.glob("results/TRANSBTS*.pt")
            for ff in f2rem:
                os.remove(ff)            
            torch.save(model.state_dict(), 'results/TRANSBTS_model_{}.pt'.format(str(epoch+epochstart)))
            f=open("results/TRANSBTS_model.txt", "a") 
            f.write('Validation epoch %d : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %f mean_recall : %f \n' % (epoch, mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
            f.close()
        elif cnt<autostop:
            cnt=cnt+1
        else:
            print("Exceed autostop limit...")
            sys.exit()
