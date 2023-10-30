from core.eval_utils import *
import torch
import torch.nn.functional as F
import SimpleITK as sitk

#ATLAS_MEAN= torch.FloatTensor(30.20063); ATLAS_SD= torch.FloatTensor(35.221165)

def znormalize(data, mean, std):
    norm_data = (data - mean) / std
    return norm_data.float()

def metric_eval(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list = 0.0
    for i_batch, sample in enumerate(tstloader):
        stack_vol = sample['t1wvol']['data'][:,:,:,:,16:-16].float().to(device)
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        gseg = sample['segvol']['data'][:,:,:,:,16:-16][0].long().squeeze(0).numpy()   
        oo = model(stack_vol)
        oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
        oseg = oseg.cpu().detach().numpy()
        metric_i = []
        for i in range(1, num_classes):
            tgt=(oseg == i)
            src=(gseg == i)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            img_itk0 = sitk.GetImageFromArray(stack_vol.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))
            prd_itk = sitk.GetImageFromArray(oseg.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(gseg.astype(np.float32))
            img_itk0.SetSpacing((1, 1, 1))
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(img_itk0, test_save_path + '/'+ str(i_batch) + "_img0.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        metric_list += np.array(metric_i)
        print(i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list = metric_list / (i_batch+1)
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    mean_precision = np.mean(metric_list, axis=0)[3]
    mean_recall = np.mean(metric_list, axis=0)[4]
    #print('Testing performance : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %mean_recall : %f' % (mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall

def metric_eval_cvitcls(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list = 0.0
    for i_batch, sample in enumerate(tstloader):
        stack_vol = sample['t1wvol']['data'][:,:,:,:,16:-16].float().to(device)
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        gseg = sample['segvol']['data'][:,:,:,:,16:-16][0].long().squeeze(0).numpy()   
        oo, _, _ = model(stack_vol)
        oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
        oseg = oseg.cpu().detach().numpy()
        metric_i = []
        for i in range(1, num_classes):
            tgt=(oseg == i)
            src=(gseg == i)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            img_itk0 = sitk.GetImageFromArray(stack_vol.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))
            prd_itk = sitk.GetImageFromArray(oseg.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(gseg.astype(np.float32))
            img_itk0.SetSpacing((1, 1, 1))
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(img_itk0, test_save_path + '/'+ str(i_batch) + "_img0.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        metric_list += np.array(metric_i)
        print(i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list = metric_list / (i_batch+1)
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    mean_precision = np.mean(metric_list, axis=0)[3]
    mean_recall = np.mean(metric_list, axis=0)[4]
    #print('Testing performance : mean_dice : %f mean_hd95 : %f mean_iou : %f mean_precision : %mean_recall : %f' % (mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall))
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall

def metric_eval_cvitcls_isles2018(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list = 0.0
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long().numpy()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            #import pdb; pdb.set_trace()
            stack_vol_=stack_vol[:,:,:,:,b].to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _ = model(stack_vol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach().numpy()
            gsum=gseg_.sum()
            if gsum >0.0:
                for i in range(1, num_classes):
                    tgt=(oseg == i)
                    src=(gseg_ == i)
                    metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        abc=np.array(metric_i)
        abc=abc.sum(0)/abc.shape[0]
        metric_list += abc
        print(i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list = metric_list / (i_batch+1)
    mean_dice = metric_list[0]
    mean_hd95 = metric_list[1]
    mean_iou = metric_list[2]
    mean_precision = metric_list[3]
    mean_recall = metric_list[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall

from core.sitk_utils import resample_to_spacing, calculate_origin_offset
import numpy as np
class Resize:
    def __init__(self, new_shape,  interpolation="linear"):
        self.new_shape = new_shape
        self.interpolation = interpolation
    def preprocess(self, image,):
        zoom_level = np.divide(self.new_shape, image.shape)
        new_spacing = np.divide((1.,1.,1.), zoom_level)
        new_data = resample_to_spacing(image, (1.,1.,1.), new_spacing, interpolation=self.interpolation)
        return new_data

def metric_eval_3dunet_isles2018(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    target_size=(256,256,4)
    r=Resize(new_shape=target_size)
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long().numpy()
        
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        stack_vol_=stack_vol.to(device)
        gseg_=gseg[0].squeeze(0)
        gseg_=r.preprocess(gseg_)
        gseg_[gseg_>0.5]=1
        gseg_[gseg_<1]=0
        oo = model(stack_vol_)
        oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
        oseg = oseg.cpu().detach().numpy()
        oseg=r.preprocess(oseg)
        oseg[oseg>0.5]=1
        oseg[oseg<1]=0
        tgt=(oseg == 1)
        src=(gseg_ == 1)
        metric_i.append((dice_score(src,tgt), hd95_score0(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg.astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_.astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(np.array(metric_i))
        print(sample['segvol']['path'][0].split("\\")[-3], " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)[0]
    metric_at_std = metric_list_at_array.std(axis=0)[0]
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall
    
def metric_eval_3dunet_isles2018_slice(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    target_size=(256,256,8)
    r=Resize(new_shape=target_size)
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long().numpy()
        
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        stack_vol_=stack_vol.to(device)
        gseg_=gseg[0].squeeze(0)
        gseg_[gseg_>0.5]=1
        gseg_[gseg_<1]=0
        oo = model(stack_vol_)
        oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
        oseg = oseg.cpu().detach().numpy()
        oseg[oseg>0.5]=1
        oseg[oseg<1]=0
        for j in range(depth):
            oseg_s=oseg[:,:,j]
            gseg_s=gseg_[:,:,j]
            tgt=(oseg_s == 1)
            src=(gseg_s == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg.astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_.astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        print(sample['segvol']['path'][0].split("\\")[-3], " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)
    metric_at_std = metric_list_at_array.std(axis=0)
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall

def metric_eval_cvitcls_isles2018slice(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            stack_vol_=stack_vol[:,:,:,:,b].to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _ = model(stack_vol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach()
            if  b==0:
                oseg_all=oseg.unsqueeze(2)
                gseg_all=gseg_.unsqueeze(2)
            else:
                oseg_all=torch.concat((oseg_all,oseg.unsqueeze(2)), axis=2)
                gseg_all=torch.concat((gseg_all,gseg_.unsqueeze(2)), axis=2)
            tgt=(oseg.numpy() == 1)
            src=(gseg_.numpy() == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg_all.numpy().astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_all.numpy().astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        path=sample['segvol']['path'][0].split("\\")
        if len(path)==1:
            path=sample['segvol']['path'][0].split("/")
        if len(path)>3:
            print(path[len(path)-3], " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
        else:
            print(" Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)
    metric_at_std = metric_list_at_array.std(axis=0)
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall


def metric_eval_cvitcls_isles2018chunk(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            if b==0:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b+1]), dim=1)
            elif b==depth-1:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b-1], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b]), dim=1)
            else:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b-1], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b+1]), dim=1)
            stack_vol_=stack_vol_.to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _ = model(stack_vol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach()
            if  b==0:
                oseg_all=oseg.unsqueeze(2)
                gseg_all=gseg_.unsqueeze(2)
            else:
                oseg_all=torch.concat((oseg_all,oseg.unsqueeze(2)), axis=2)
                gseg_all=torch.concat((gseg_all,gseg_.unsqueeze(2)), axis=2)
            tgt=(oseg.numpy() == 1)
            src=(gseg_.numpy() == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg_all.numpy().astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_all.numpy().astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        print(sample['segvol']['path'][0].split("\\")[-3], " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)[0]
    metric_at_std = metric_list_at_array.std(axis=0)[0]
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall

def metric_eval_cvitcls_isles2018chunk2(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        zstack=torch.zeros_like(stack_vol[:,:,:,:,0])
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            if b==0:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b+1]), dim=0)
            elif b==depth-1:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b-1], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b]), dim=0)
            else:
                stack_vol_=torch.concat((stack_vol[:,:,:,:,b-1], stack_vol[:,:,:,:,b],stack_vol[:,:,:,:,b+1]), dim=0)
            stack_vol_=stack_vol_.to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _ = model(stack_vol_[0].unsqueeze(0), stack_vol_[1].unsqueeze(0), stack_vol_[2].unsqueeze(0))
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach()
            if  b==0:
                oseg_all=oseg.unsqueeze(2)
                gseg_all=gseg_.unsqueeze(2)
            else:
                oseg_all=torch.concat((oseg_all,oseg.unsqueeze(2)), axis=2)
                gseg_all=torch.concat((gseg_all,gseg_.unsqueeze(2)), axis=2)
            tgt=(oseg.numpy() == 1)
            src=(gseg_.numpy() == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg_all.numpy().astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_all.numpy().astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        print(" Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)
    metric_at_std = metric_list_at_array.std(axis=0)
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall


def metric_eval_cvitcls_isles2018chunk2b(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list = 0.0
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long().numpy()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            if b==0:
                cbvvol_=torch.concat((cbvvol[:,:,:,:,b], cbvvol[:,:,:,:,b],cbvvol[:,:,:,:,b+1]), dim=1)
                cbfvol_=torch.concat((cbfvol[:,:,:,:,b], cbfvol[:,:,:,:,b],cbfvol[:,:,:,:,b+1]), dim=1)
                mttvol_=torch.concat((mttvol[:,:,:,:,b], mttvol[:,:,:,:,b],mttvol[:,:,:,:,b+1]), dim=1)
                tmaxvol_=torch.concat((tmaxvol[:,:,:,:,b], tmaxvol[:,:,:,:,b],tmaxvol[:,:,:,:,b+1]), dim=1)
            elif b==depth-1:
                cbvvol_=torch.concat((cbvvol[:,:,:,:,b-1], cbvvol[:,:,:,:,b],cbvvol[:,:,:,:,b]), dim=1)
                cbfvol_=torch.concat((cbfvol[:,:,:,:,b-1], cbfvol[:,:,:,:,b],cbfvol[:,:,:,:,b]), dim=1)
                mttvol_=torch.concat((mttvol[:,:,:,:,b-1], mttvol[:,:,:,:,b],mttvol[:,:,:,:,b]), dim=1)
                tmaxvol_=torch.concat((tmaxvol[:,:,:,:,b-1], tmaxvol[:,:,:,:,b],tmaxvol[:,:,:,:,b]), dim=1)
            else:
                cbvvol_=torch.concat((cbvvol[:,:,:,:,b-1], cbvvol[:,:,:,:,b],cbvvol[:,:,:,:,b+1]), dim=1)
                cbfvol_=torch.concat((cbfvol[:,:,:,:,b-1], cbfvol[:,:,:,:,b],cbfvol[:,:,:,:,b+1]), dim=1)
                mttvol_=torch.concat((mttvol[:,:,:,:,b-1], mttvol[:,:,:,:,b],mttvol[:,:,:,:,b+1]), dim=1)
                tmaxvol_=torch.concat((tmaxvol[:,:,:,:,b-1], tmaxvol[:,:,:,:,b],tmaxvol[:,:,:,:,b+1]), dim=1)
            cbvvol_=cbvvol_.to(device)
            cbfvol_=cbfvol_.to(device)
            mttvol_=mttvol_.to(device)
            tmaxvol_=tmaxvol_.to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _ = model(cbvvol_,cbfvol_,mttvol_,tmaxvol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach().numpy()
            gsum=gseg_.sum()
            if gsum >0.0:
                for i in range(1, num_classes):
                    tgt=(oseg == i)
                    src=(gseg_ == i)
                    metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        abc=np.array(metric_i)
        abc=abc.sum(0)/abc.shape[0]
        metric_list += abc
        print(i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list = metric_list / (i_batch+1)
    mean_dice = metric_list[0]
    mean_hd95 = metric_list[1]
    mean_iou = metric_list[2]
    mean_precision = metric_list[3]
    mean_recall = metric_list[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall

def metric_eval_fcnn_isles2018(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            stack_vol_=stack_vol[:,:,:,:,b].to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo = model(stack_vol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach()
            if  b==0:
                oseg_all=oseg.unsqueeze(2)
                gseg_all=gseg_.unsqueeze(2)
            else:
                oseg_all=torch.concat((oseg_all,oseg.unsqueeze(2)), axis=2)
                gseg_all=torch.concat((gseg_all,gseg_.unsqueeze(2)), axis=2)
            tgt=(oseg.numpy() == 1)
            src=(gseg_.numpy() == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg_all.numpy().astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_all.numpy().astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        print(sample['segvol']['path'][0].split("\\")[-3], " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)
    metric_at_std = metric_list_at_array.std(axis=0)
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall

def metric_eval_fcnn_isles2018_deep(model, tstloader, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list_at=[]
    for i_batch, sample in enumerate(tstloader):
        cbvvol = sample['cbvvol']['data']
        cbfvol = sample['cbfvol']['data']
        mttvol = sample['mttvol']['data']
        tmaxvol = sample['tmaxvol']['data']
        stack_vol=torch.concat((cbvvol,cbfvol,mttvol,tmaxvol),dim=1)
        depth=tmaxvol.shape[4]
        gseg = sample['segvol']['data'].long()
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        metric_i = []
        for b in range(depth):
            stack_vol_=stack_vol[:,:,:,:,b].to(device)
            gseg_=gseg[:,:,:,:,b][0].squeeze(0)
            oo, _, _, _ = model(stack_vol_)
            oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
            oseg = oseg.cpu().detach()
            if  b==0:
                oseg_all=oseg.unsqueeze(2)
                gseg_all=gseg_.unsqueeze(2)
            else:
                oseg_all=torch.concat((oseg_all,oseg.unsqueeze(2)), axis=2)
                gseg_all=torch.concat((gseg_all,gseg_.unsqueeze(2)), axis=2)
            tgt=(oseg.numpy() == 1)
            src=(gseg_.numpy() == 1)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            prd_itk = sitk.GetImageFromArray(np.transpose(oseg_all.numpy().astype(np.float32), (2,1,0)))
            lab_itk = sitk.GetImageFromArray(np.transpose(gseg_all.numpy().astype(np.float32), (2,1,0)))            
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        at_arry = np.array(metric_i)
        metric_list_at.append(at_arry.mean(0))
        print( " Index: ", i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list_at_array=np.array(metric_list_at)
    metric_at_mean = metric_list_at_array.mean(axis=0)
    metric_at_std = metric_list_at_array.std(axis=0)
    mean_dice = metric_at_mean[0]
    mean_hd95 = metric_at_mean[1]
    mean_iou = metric_at_mean[2]
    mean_precision = metric_at_mean[3]
    mean_recall = metric_at_mean[4]
    std_dice = metric_at_std[0]
    std_hd95 = metric_at_std[1]
    std_iou = metric_at_std[2]
    std_precision = metric_at_std[3]
    std_recall = metric_at_std[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall, std_dice, std_hd95, std_iou, std_precision, std_recall

def metric_eval_cvitcls_sl(model, tstloader, sl_dict, norm=False, mean=0., std=1., device='cpu', num_classes=1, save_image=False, test_save_path='./results/images'):
    model.eval()
    metric_list = 0.0
    sl_key=list(sl_dict.keys())
    
    for i_batch, sample in enumerate(tstloader):
        stack_vol = sample['t1wvol']['data'][:,:,:,:,16:-16].float().to(device)
        if norm:
            stack_vol = znormalize(stack_vol, mean, std)
        gseg = sample['segvol']['data'][:,:,:,:,16:-16][0].long().squeeze(0).numpy()   
        oo, _, _ = model(stack_vol)
        oseg = torch.argmax(F.softmax(oo, dim=1), dim=1).squeeze(0)
        oseg = oseg.cpu().detach().numpy()
        oseg_sl= oseg[:, sl_dict[sl_key[i_batch]]]
        gseg_sl= gseg[:, sl_dict[sl_key[i_batch]]]
        metric_i = []
        for i in range(1, num_classes):
            tgt=(oseg_sl == i)
            src=(gseg_sl == i)
            metric_i.append((dice_score(src,tgt), hd95_score(src,tgt), IOU(src,tgt), precision(src,tgt), recall(src,tgt)) )
        if save_image:
            img_itk0 = sitk.GetImageFromArray(stack_vol.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))
            prd_itk = sitk.GetImageFromArray(oseg.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(gseg.astype(np.float32))
            img_itk0.SetSpacing((1, 1, 1))
            prd_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ str(i_batch) + "_pred.nii.gz")
            sitk.WriteImage(img_itk0, test_save_path + '/'+ str(i_batch) + "_img0.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ str(i_batch)  + "_gt.nii.gz")
        metric_list += np.array(metric_i)
        print(i_batch," DICE: ", np.mean(metric_i, axis=0)[0], "HD95: ", np.mean(metric_i, axis=0)[1], "IOU: ", np.mean(metric_i, axis=0)[2], "Precision: ", np.mean(metric_i, axis=0)[3], "Recall: ", np.mean(metric_i, axis=0)[4])
    metric_list = metric_list / (i_batch+1)
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    mean_precision = np.mean(metric_list, axis=0)[3]
    mean_recall = np.mean(metric_list, axis=0)[4]
    return mean_dice, mean_hd95, mean_iou, mean_precision, mean_recall
