import cv2, time
import torch
import numpy as np
from sklearn import metrics
from medpy import metric as medmetric
from core.data_utils import save_numpy_to_nii, save_numpy, get_n_chunk

def hd95_score0(src, tgt):
    if tgt.sum() > 0 and src.sum()>0:
        hd95 = medmetric.binary.hd95(tgt, src)
        return hd95
    elif tgt.sum() > 0 and src.sum()==0:
        return 0
    else:
        return 0

def hd95_score1(src_all, tgt_all):
    assert len(src_all.shape)==3
    src_all=np.transpose(src_all,(2,0,1))
    tgt_all=np.transpose(tgt_all,(2,0,1))
    hd95 = 0.0
    for j in range(src_all.shape[0]):
        tgt=(tgt_all[j]==1)
        src=(src_all[j]==1)
        if tgt.sum() > 0 or src.sum() > 0:
            if src.sum() == 0:
                idx=int(np.ma.masked_invalid(np.array([np.where(tgt[i]==1)[0].mean() for i in range(tgt.shape[0])])).mean())
                tgtt=np.transpose(tgt, (1,0))
                idy=int(np.ma.masked_invalid(np.array([np.where(tgtt[i]==1)[0].mean() for i in range(tgtt.shape[0])])).mean())
                src[idx,idy]=True
            if tgt.sum() == 0:
                idx=int(np.ma.masked_invalid(np.array([np.where(src[i]==1)[0].mean() for i in range(src.shape[0])])).mean())
                srct=np.transpose(src, (1,0))
                idy=int(np.ma.masked_invalid(np.array([np.where(srct[i]==1)[0].mean() for i in range(srct.shape[0])])).mean())
                tgt[idx,idy]=True
            tmp = medmetric.binary.hd95(tgt, src)
        else:
            tmp=0.0
        hd95 = hd95 + tmp
    return hd95/src_all.shape[0]

def hd95_score(src, tgt):
    assert len(src.shape)==2
    hd95 = 0.0
    if tgt.sum() > 0 or src.sum() > 0:
        if src.sum() == 0:
            idx=int(np.ma.masked_invalid(np.array([np.where(tgt[i]==1)[0].mean() for i in range(tgt.shape[0])])).mean())
            tgtt=np.transpose(tgt, (1,0))
            idy=int(np.ma.masked_invalid(np.array([np.where(tgtt[i]==1)[0].mean() for i in range(tgtt.shape[0])])).mean())
            src[idx,idy]=True
        if tgt.sum() == 0:
            idx=int(np.ma.masked_invalid(np.array([np.where(src[i]==1)[0].mean() for i in range(src.shape[0])])).mean())
            srct=np.transpose(src, (1,0))
            idy=int(np.ma.masked_invalid(np.array([np.where(srct[i]==1)[0].mean() for i in range(srct.shape[0])])).mean())
            tgt[idx,idy]=True
        tmp = medmetric.binary.hd95(tgt, src)
    else:
        tmp=0.0
    return tmp

def dice_score(src, tgt):
    '''
    Calculate Dice Score

    Arg(s):
        src: numpy array of predicted segmentation
        tgt: numpy array of ground truth segmentation
    Returns:
        float : dice score ( 2 * (A intersect B)/(|A| + |B|))
    '''

    intersection = np.logical_and(src, tgt)
    total = src.sum() + tgt.sum()
    if total == 0:  
        return 0.0
    return 2 * intersection.sum() / total

def IOU(src, tgt):
    '''
    Calculate Intersection Over Union (IOU)

    Arg(s):
        src: numpy
            numpy array of predicted segmentation
        tgt: numpy
            numpy array of ground truth segmentation
    Returns:
        float : intersection over union ((A intersect B)/ (A U B))
    '''

    intersection = np.logical_and(src, tgt)
    union = np.logical_or(src, tgt)
    if union.sum() == 0: 
        return 0.0
    return intersection.sum() / union.sum()

def per_class_iou(hist):
    '''
    Calculate IOU per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        iou for the ith class.
    '''

    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is ground-truth or predicted:
    union = hist.sum(1) + hist.sum(0) - nb_of_tp

    # compute iou:
    iou_per_class = nb_of_tp / union

    return iou_per_class

def per_class_dice(hist):
    '''
    Calculate Dice per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        dice for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # sum of times a class is ground-truth or predicted:
    denom = hist.sum(1) + hist.sum(0)

    # compute dice:
    dice_per_class = 2*nb_of_tp / denom

    return dice_per_class

def perclass2mean(per_class_stat):
    '''
    Calculate means per class (non-lesion, lesion, overall)

    Arg(s):
        per_class_stat : numpy
            2D array (N x 2) where columns represent statistic for non-lesion and lesion classes
    Returns:
        tuple[float] : mean values by column (non_lesion_mean, lesion_mean, overall_mean)
    '''
    non_lesion_mean = np.nanmean(per_class_stat[:, 0])
    lesion_mean = np.nanmean(per_class_stat[:, 1])
    overall_mean = np.nanmean(per_class_stat)

    return non_lesion_mean, lesion_mean, overall_mean

def compute_prediction_hist(label, pred, num_classes):
    '''
    Given labels, predictions, and the number of classes, compute a histogram summary

    Arg(s):
        label : numpy
            1D numpy array of ground truth labels
        pred : numpy
            1D numpy array of predicted labels with the same size as label
        num_classes : int
            number of classes
    Returns:
        numpy : 2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class.
    '''
    # Sanity checks
    assert len(label) == len(pred), (len(label), len(pred))
    assert label.ndim == 1, label.ndim
    assert pred.ndim == 1, pred.ndim

    '''
    mask is a boolean vector of length len(label) to ignore the invalid pixels
    e.g. when there is an ignored class which is assigned to 255.
    '''
    mask = (label >= 0) & (label < num_classes)

    '''
    label_pred_1d is a 1D vector for valid pixels, where gt labels are modulated
    with num_classes to store them in the rows of the prediction histogram.
    Goal is to encode number of times each (label, pred) pair is seen.
    '''
    label_pred_1d = num_classes * label[mask].astype(int) + pred[mask]

    # convert set of (label, pred) pairs into a 1D histogram of size num_classes**2:
    hist_1d = np.bincount(label_pred_1d, minlength=num_classes**2)

    # convert 1d histogram to 2d histogram:
    hist_2d = hist_1d.reshape(num_classes, num_classes)
    assert hist_2d.shape[0] == num_classes, (hist_2d.shape[0], num_classes)
    assert hist_2d.shape[1] == num_classes, (hist_2d.shape[1], num_classes)

    return hist_2d

def precision(src, tgt):
    '''
    Calculate precision

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : precision = (true positives / (true positives + false positives))
    '''
    src_positives = np.sum(src)   # = true positives + false positives
    true_positives = np.sum(np.logical_and(src, tgt))
    if src_positives == 0:
        return 0.0

    return true_positives / src_positives

def per_class_precision(hist):
    '''
    Calculate precision per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        precision for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is predicted:
    nb_pred = hist.sum(0)

    # compute precision:
    precision_per_class = nb_of_tp / nb_pred

    return precision_per_class

def recall(src, tgt):
    '''
    Calculate recall

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : recall = (true positives / (true positives + false negatives))
    '''
    true_positives = np.sum(np.logical_and(src, tgt))
    inverse_src = np.logical_not(src)

    # false negative is prediction labeled as not lesion but actually is lesion
    false_negatives = np.sum(np.logical_and(inverse_src, tgt))
    if (true_positives + false_negatives) == 0:
        return 0.0

    return true_positives / (true_positives + false_negatives)

def per_class_recall(hist):
    '''
    Calculate recall per class based on histogram

    Arg(s):
        hist : numpy
            2D numpy array of size num_classes X num_classes whose (i,j)-th
            element is the number of times jth class is predicted for what supposed
            to be ith class as returned by compute_prediction_hist().
    Returns:
        numpy : 1D numpy array of size num_classes. ith element is the
        recall for the ith class.
    '''
    # number of true positives for each class:
    nb_of_tp = np.diag(hist)

    # number of times a class is the ground truth:
    nb_gt = hist.sum(1)

    # compute recall:
    recall_per_class = nb_of_tp / nb_gt

    return recall_per_class

def accuracy(src, tgt):
    '''
    Calculate accuracy

    Arg(s):
        src: numpy
            numpy array of predicted segmentation
        tgt: numpy
            numpy array of ground truth segmentation
    Returns:
        float : accuracy (|src - tgt| / tgt)
    '''
    true_positives = np.sum(np.logical_and(src, tgt))
    inverse_src = np.logical_not(src)
    inverse_tgt = np.logical_not(tgt)
    true_negatives = np.sum(np.logical_and(inverse_src, inverse_tgt))

    total = np.sum(np.ones_like(tgt))
    return (true_positives + true_negatives) / total

def specificity(src, tgt):
    '''
    Calculate specificity

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : specificty = (true negatives / (true negatives + false positives))
    '''
    inverse_src = np.logical_not(src)
    inverse_tgt = np.logical_not(tgt)

    true_negatives = np.sum(np.logical_and(inverse_src, inverse_tgt))
    false_positives = np.sum(np.logical_and(src, inverse_tgt))

    # Check for divide by 0
    if true_negatives + false_positives == 0:
        return 0.0

    return true_negatives / (true_negatives + false_positives)

def f1(src, tgt):
    '''
    Calculate f1

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : f1 = 2 * precision * recall / (precision + recall)
    '''
    precision_metric = precision(src, tgt)
    recall_metric = recall(src, tgt)

    if precision_metric + recall_metric == 0:
        return 0.0

    return (2 * precision_metric * recall_metric) / (precision_metric + recall_metric)

def auc_roc(src, tgt):
    '''
    Calculate area under the curve of ROC curve

    Arg(s):
        src : numpy
            numpy array of predicted segmentation
        tgt : numpy
            numpy array of ground truth segmentation
    Returns:
        float : AUC of ROC
    '''
    # sklearn's function takes in flattened 1D arrays
    assert src.shape == tgt.shape
    if len(src.shape) != 1:
        src = src.flatten()
        tgt = tgt.flatten()
    # target cannot be all same element
    if len(np.unique(tgt)) == 1:
        return None
    return metrics.roc_auc_score(tgt, src)
