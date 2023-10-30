import os
import random
from PIL import Image
import numpy as np
import nibabel as nib
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from pathlib import Path
import torchio as tio
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    Crop,
    Flip,
    HistogramStandardization,
    OneOf,
    Compose,
)

import matplotlib.pyplot as plt
import tempfile
import SimpleITK as sitk
from torch.utils.data import Dataset

class isles2018dataset(Dataset):
    def __init__(self, path, transform=None):
        self.path=path
        self.imageCBV=glob.glob(path+"/*/*/*CBV*.nii")
        self.imageCBF=glob.glob(path+"/*/*/*CBF*.nii")
        self.imageTmax=glob.glob(path+"/*/*/*Tmax*.nii")
        self.imageMTT=glob.glob(path+"/*/*/*MTT*.nii")
        self.imageOT=glob.glob(path+"/*/*/*OT*.nii")
        self.transform = transform
    def __getitem__(self, index):
        cbv=torch.tensor(nib.load(self.imageCBV[index]).get_fdata()).unsqueeze(0)
        cbf=torch.tensor(nib.load(self.imageCBF[index]).get_fdata()).unsqueeze(0)
        mtt=torch.tensor(nib.load(self.imageMTT[index]).get_fdata()).unsqueeze(0)
        tmax=torch.tensor(nib.load(self.imageTmax[index]).get_fdata()).unsqueeze(0)
        ot=torch.tensor(nib.load(self.imageOT[index]).get_fdata())
        concat=torch.cat((cbv,cbf,mtt,tmax))
        if transform:
            concat=transform(concat)
        return concat, ot
    def __len__(self):
        return len(self.imageCBV)

def show_nifti(image_path_or_image, colormap='gray'):
    try:
        from niwidgets import NiftiWidget
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            widget = NiftiWidget(image_path_or_image)
            widget.nifti_plotter(colormap=colormap)
    except Exception:
        if isinstance(image_path_or_image, nib.AnalyzeImage):
            nii = image_path_or_image
        else:
            image_path = image_path_or_image
            nii = nib.load(str(image_path))
        k = int(nii.shape[-1] / 2)
        plt.imshow(nii.dataobj[..., k], cmap=colormap)

def show_subject(subject, image_name, label_name=None):
    if label_name is not None:
        subject = copy.deepcopy(subject)
        affine = subject[label_name].affine
        label_image = subject[label_name].as_sitk()
        label_image = sitk.Cast(label_image, sitk.sitkUInt8)
        border = sitk.BinaryContour(label_image)
        border_array, _ = tio.utils.sitk_to_nib(border)
        border_tensor = torch.from_numpy(border_array)[0]
        image_tensor = subject[image_name].data[0]
        image_tensor[border_tensor > 0.5] = image_tensor.max()
    with tempfile.NamedTemporaryFile(suffix='.nii') as f:
        subject[image_name].save(f.name)
        show_nifti(f.name)

def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)
    
def make_sublist_isles2018(data_path):
        data_path=Path(data_path)
        cbv_list = sorted(data_path.glob("*/*/*CBV*.niiresave.nii"))
        cbf_list = sorted(data_path.glob("*/*/*CBF*.niiresave.nii"))
        mtt_list = sorted(data_path.glob("*/*/*MTT*.niiresave.nii"))
        tmax_list = sorted(data_path.glob("*/*/*Tmax*.niiresave.nii"))
        seg_vol_list = sorted(data_path.glob("*/*/*OT*.niiresave.nii"))
        sub_list=[]
        for (cbv, cbf, mtt, tmax, seg) in zip(cbv_list, cbf_list, mtt_list, tmax_list, seg_vol_list):
                sub = tio.Subject(
                        cbvvol = tio.ScalarImage(cbv),
                        cbfvol = tio.ScalarImage(cbf),
                        mttvol = tio.ScalarImage(mtt),
                        tmaxvol = tio.ScalarImage(tmax),
                        segvol = tio.LabelMap(seg),
                )
                sub_list.append(sub)
        #dataset = tio.SubjectsDataset(sub_list)
        #import pdb; pdb.set_trace()
        return sub_list

def make_sublist_isles2018_2(data_path):
        data_path=Path(data_path)
        cbv_list = sorted(data_path.glob("*/*CBV*.niiresave.nii"))
        cbf_list = sorted(data_path.glob("*/*CBF*.niiresave.nii"))
        mtt_list = sorted(data_path.glob("*/*MTT*.niiresave.nii"))
        tmax_list = sorted(data_path.glob("*/*Tmax*.niiresave.nii"))
        seg_vol_list = sorted(data_path.glob("*/*OT*.niiresave.nii"))
        sub_list=[]
        for (cbv, cbf, mtt, tmax, seg) in zip(cbv_list, cbf_list, mtt_list, tmax_list, seg_vol_list):
                sub = tio.Subject(
                        cbvvol = tio.ScalarImage(cbv),
                        cbfvol = tio.ScalarImage(cbf),
                        mttvol = tio.ScalarImage(mtt),
                        tmaxvol = tio.ScalarImage(tmax),
                        segvol = tio.LabelMap(seg),
                )
                sub_list.append(sub)
        return sub_list
def make_sublist_atlas(data_path):
        data_path=Path(data_path)
        t1w_vol_list = sorted(data_path.glob("*T1w*.nii.gz"))
        seg_vol_list = sorted(data_path.glob("*mask*.nii.gz"))
        sub_list=[]
        for (t1w, seg) in zip(t1w_vol_list, seg_vol_list):
                sub = tio.Subject(
                        t1wvol = tio.ScalarImage(t1w),
                        segvol = tio.LabelMap(seg),
                )
                sub_list.append(sub)
        #dataset = tio.SubjectsDataset(sub_list)
        return sub_list

def make_sublist_atlasr12(data_path):
        data_path=Path(data_path)
        t1w_vol_list = sorted(data_path.glob("*t1w_stx*.nii.gz"))
        seg_vol_list = sorted(data_path.glob("*LesionSmooth_stx*.nii.gz"))
        sub_list=[]
        for (t1w, seg) in zip(t1w_vol_list, seg_vol_list):
                sub = tio.Subject(
                        t1wvol = tio.ScalarImage(t1w),
                        segvol = tio.LabelMap(seg),
                )
                sub_list.append(sub)
        return sub_list

import glob
def make_sublist_atlasr12_sl(data_path, sl_dict):
        data_path0=Path(data_path)
        t1w_vol_list = sorted(data_path0.glob("*t1w_stx*.nii.gz"))
        seg_vol_list = sorted(data_path0.glob("*LesionSmooth_stx*.nii.gz"))
        seg_list = sorted(glob.glob(data_path+"/*LesionSmooth_stx*.nii.gz"))
        i=0
        for k in sl_dict.keys():
            if k not in seg_list[i]:
                print("Error!")
                exit()
            i=i+1
        sub_list=[]
        for (t1w, seg) in zip(t1w_vol_list, seg_vol_list):
                sub = tio.Subject(
                        t1wvol = tio.ScalarImage(t1w),
                        segvol = tio.LabelMap(seg),
                )
                sub_list.append(sub)
        return sub_list

def transformisles(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.75,1.5,0.75,1.5,1,1), degrees=(0,0,20), image_interpolation='nearest'),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   
        validation_transform = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        nonorm_transform = Compose([
            ToCanonical(),
        ])
        nonorm_transform_aug = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.75,1.5,0.75,1.5,1,1), degrees=(0,0,20), image_interpolation='nearest'),            
        ])
        if train_mode==1:
                transform = training_transform
        elif train_mode==2:
                transform = nonorm_transform
        elif train_mode==3:
                transform = nonorm_transform_aug
        else:
                transform = validation_transform
        return transform

class RandomCrop(object):
    def __init__(self):
        self.W=40
        self.H=34
        self.D=8
        self.maxWs=32
        self.maxHs=64
        self.orgW=240
        self.orgH=240
        self.orgD=155
        self.newWHD=128
    def __call__(self, sample):
        w=np.random.randint(0, self.maxWs) + self.W
        h=np.random.randint(0, self.maxHs) + self.H
        d=self.D
        #print(w, " " ,240-128-w, " " , h, " " , 240-128-h)
        a=Crop((w, self.orgW-self.newWHD-w, h, self.orgH-self.newWHD-h, d, self.orgD-self.newWHD-d))(sample)
        #a=Crop((128, 128, 128))(sample)
        return a

class RandomCrop192(object):
    def __init__(self):
        self.maxWs=(240-192)
        self.maxHs=(240-192)
        self.maxDs=40
        self.orgW=240
        self.orgH=240
        self.orgD=155
        self.tarW=192
        self.tarH=192
        self.tarD=96
    def __call__(self, sample):
        w=np.random.randint(0, self.maxWs) 
        h=np.random.randint(0, self.maxHs)
        d=np.random.randint(20, self.maxDs)
        #print("w,h,d: ", w,h,d)
        a=Crop((w,self.orgW-self.tarW-w, h,self.orgH-self.tarH-h, d,self.orgD-self.tarD-d))(sample)
        return a

class RandomCrop192x64(object):
    def __init__(self):
        self.maxWs=(240-192)
        self.maxHs=(240-192)
        self.maxDs=120-64
        self.orgW=240
        self.orgH=240
        self.orgD=155
        self.tarW=192
        self.tarH=192
        self.tarD=64
    def __call__(self, sample):
        w=np.random.randint(0, self.maxWs) 
        h=np.random.randint(0, self.maxHs)
        d=np.random.randint(30, self.maxDs)
        #print("w,h,d: ", w,h,d)
        a=Crop((w,self.orgW-self.tarW-w, h,self.orgH-self.tarH-h, d,self.orgD-self.tarD-d))(sample)
        return a

     
class FixedCrop(object):
    def __init__(self):
        self.W=40
        self.H=34
        self.D=8
        self.maxWs=32
        self.maxHs=64
        self.orgW=240
        self.orgH=240
        self.orgD=155
        self.newWHD=128
    def __call__(self, sample):
        w=int(self.maxWs/2) + self.W
        h=int(self.maxHs/2) + self.H
        d=self.D
        #print(w, " " ,240-128-w, " " , h, " " , 240-128-h)
        a=Crop((w, self.orgW-self.newWHD-w, h, self.orgH-self.newWHD-h, d, self.orgD-self.newWHD-d))(sample)
        return a


def transform128(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(degrees=5, center='image', isotropic=True, image_interpolation='linear'),
            FixedCropWin(),
            Resample((1.03125, 1.296875, 0.8203125)),
            RandomAffine(scales=(0.97, 1.03), center='image', isotropic=True, image_interpolation='linear'),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   
        validation_transform = Compose([
            ToCanonical(),
            FixedCropWin(),
            Resample((1.03125, 1.296875, 0.8203125)),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        else:
                transform = validation_transform
        return transform

def transform224x80(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            FixedCrop224x80(8, 8, 36),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   
        validation_transform = Compose([
            ToCanonical(),
            FixedCrop224x80(8, 8, 28),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        else:
                transform = validation_transform
        return transform

def transform192(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            FixedCrop192x96(24, 24, 28),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   

        validation_transform = Compose([
            ToCanonical(),
            FixedCrop192x96(24, 24, 28),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        else:
                transform = validation_transform
        return transform

def transform128128128(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.6, 1.5), degrees=25, center='image', isotropic=True, p=0.3),
            CropOrPad((128, 128, 128)),
            ZNormalization(),
        ])
        training_transform_n = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='linear'),
            CropOrPad((128, 128, 128)),
            ZNormalization(),
        ])   
        training_transform_100 = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=25, center='image', isotropic=True, image_interpolation='linear'),
            CropOrPad((128, 128, 128)),
            ZNormalization(),
        ])
        validation_transform = Compose([
            ToCanonical(),
            CropOrPad((128, 128, 128)),
            ZNormalization(),
        ])
        validation_transform_2a = Compose([
            ToCanonical(),
            Crop128x128x128(50, 22, 16),
            #ZNormalization(),
        ])
        validation_transform_2b = Compose([
            ToCanonical(),
            Crop128x128x128(70, 22, 16),
            #ZNormalization(),
        ])        
        validation_transform_2c = Compose([
            ToCanonical(),
            Crop128x128x128(50, 72, 16),
            #ZNormalization(),
        ])
        validation_transform_2d = Compose([
            ToCanonical(),
            Crop128x128x128(70, 72, 16),
            #ZNormalization(),
        ])
        if train_mode==1:
                transform = training_transform
        elif train_mode==0:
                transform = training_transform_n
        elif train_mode==2:
                transform = validation_transform_2a
        elif train_mode==3:
                transform = validation_transform_2b
        elif train_mode==4:
                transform = validation_transform_2c
        elif train_mode==5:
                transform = validation_transform_2d
        else:
                transform = validation_transform
                #transform=None
        return transform

class Crop128x128x128(object):
    def __init__(self, w, h, d):
        self.W=w
        self.H=h
        self.D=d
        self.orgW=240
        self.orgH=240
        self.orgD=155
        self.tarW=128
        self.tarH=128
        self.tarD=128
    def __call__(self, sample):
        a=Crop((self.W, self.orgW-self.tarW-self.W, self.H, self.orgH-self.tarH-self.H, self.D, self.orgD-self.tarD-self.D))(sample)
        return a

def transform160160192(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   

        training_transform_s = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(1., 1.1), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        
        validation_transform = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        elif train_mode==2:
                transform = training_transform_s
        else:
                transform = validation_transform
        return transform

def transform160160192wCrop(train_mode=1, crop=(50,40,70)):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            Crop60x60x68(crop[0], crop[1], crop[2]),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   

        training_transform_2 = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            Crop96x96x96(crop[0], crop[1], crop[2]),
            ZNormalization(masking_method=ZNormalization.mean),
        ])

        training_transform_r12 = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(1., 1.1), degrees=15, center='image', isotropic=True, image_interpolation='linear'),
            ZNormalization(),
        ])

        training_transform_r12_nocropnorm = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(1., 1.1), degrees=15, center='image', isotropic=True, image_interpolation='linear', p=0.75),
        ])
        training_transform_r12_nocropnorm2 = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.9, 1.25), degrees=25, center='image', isotropic=True, image_interpolation='linear', p=0.3),
        ])        
        validation_nocrop_transform = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        validation_nocropnorm_transform = Compose([
            ToCanonical(),
        ])        
        validation_transform = Compose([
            ToCanonical(),
            Crop96x96x96(crop[0], crop[1], crop[2]),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        elif train_mode==2:
                transform = training_transform_2
        elif train_mode==3:
                transform = training_transform_r12
        elif train_mode==4:
                transform = training_transform_r12_nocropnorm
        elif train_mode==5:
                transform = validation_nocrop_transform
        elif train_mode==6:
                transform = validation_nocropnorm_transform
        elif train_mode==7:
                transform = training_transform_r12_nocropnorm2
        else:
                transform = validation_transform
        return transform

class Crop60x60x68(object):
    def __init__(self, w, h, d):
        self.W=w
        self.H=h
        self.D=d
        self.orgW=160
        self.orgH=160
        self.orgD=192
        self.tarW=20*3
        self.tarH=20*3
        self.tarD=24*3
    def __call__(self, sample):
        a=Crop((self.W,self.orgW-self.tarW-self.W,self.H,self.orgH-self.tarH-self.H,self.D,self.orgD-self.tarD-self.D))(sample)
        return a

class Crop96x96x96(object):
    def __init__(self, w, h, d):
        self.W=w
        self.H=h
        self.D=d
        self.orgW=160
        self.orgH=160
        self.orgD=192
        self.tarW=96
        self.tarH=96
        self.tarD=96
    def __call__(self, sample):
        a=Crop((self.W,self.orgW-self.tarW-self.W,self.H,self.orgH-self.tarH-self.H,self.D,self.orgD-self.tarD-self.D))(sample)
        return a

class FixedCrop60x60x68(object):
    def __init__(self, w, h, d):
        self.W=w
        self.H=h
        self.D=d
        self.orgW=60
        self.orgH=60
        self.orgD=68
        self.tarW=40
        self.tarH=40
        self.tarD=48
    def __call__(self, sample):
        a=Crop((self.W,self.orgW-self.tarW-self.W,self.H,self.orgH-self.tarH-self.H,self.D,self.orgD-self.tarD-self.D))(sample)
        return a

def transform224(train_mode=1):
        training_transform = Compose([
            ToCanonical(),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.98, 1.02), degrees=5, center='image', isotropic=True, image_interpolation='nearest'),
            FixedCrop224(8, 8, 28),
            ZNormalization(masking_method=ZNormalization.mean),
        ])   
        validation_transform = Compose([
            ToCanonical(),
            FixedCrop224(8, 8, 28),
            ZNormalization(masking_method=ZNormalization.mean),
        ])
        if train_mode==1:
                transform = training_transform
        else:
                transform = validation_transform
        return transform

def transform(train_mode=1, w=None, h=None, d=None):
        training_transform_resize = Compose([
            Resample((1.875,1.875,1)),
            CropOrPad((128, 128, 128), padding_mode='reflect'),
            ZNormalization(masking_method=ZNormalization.mean),
            RandomFlip(axes=(0,)),
        ])
        training_transform_srot = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
            RandomAffine(scales=(0.92, 1.1), degrees=15, center='image', isotropic=True, image_interpolation='nearest'),
            RandomCrop(),
            RandomFlip(axes=(0,)),
        ])
        training_transform_rot = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
            RandomAffine(degrees=10, center='image', image_interpolation='nearest'),
            RandomCrop(),
            RandomFlip(axes=(0,)),
        ])
        training_transform = Compose([
            ToCanonical(),
            ZNormalization(masking_method=ZNormalization.mean),
            RandomCrop(),
            RandomFlip(axes=(0,)),
        ])          
        validation_transform_resize = Compose([
            ToCanonical(),
            Resample((1.875,1.875,1)),
            CropOrPad((128, 128, 128), padding_mode='reflect'),
            ZNormalization(masking_method=ZNormalization.mean),
        ])  
        validation_transform = Compose([
            ToCanonical(),
            FixedCrop(),
            ZNormalization(masking_method=ZNormalization.mean),
        ])      
        if train_mode==1:
                transform = training_transform
        elif train_mode==2:
                transform = training_transform_rot
        elif train_mode==3:
                transform = training_transform_srot
        else:
                transform = validation_transform
        return transform
