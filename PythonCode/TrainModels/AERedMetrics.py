import numpy as np
import cv2
import pandas as pd
from skimage import io
import glob
import random
from QualityMetrics import AUCMetrics

def redness(original_image, reconstructed_image):
    original_image = np.float32(original_image)
    reconstructed_image = np.float32(reconstructed_image)
    redness_original = original_image[:,:,0] - (original_image[:,:,1] + original_image[:,:,2]) / 2
    redness_reconstructed = reconstructed_image[:,:,0] - (reconstructed_image[:,:,1] + reconstructed_image[:,:,2]) / 2
    return np.mean((redness_original - redness_reconstructed) ** 2)

def fractionOfRed(original_image, reconstructed_image):
    original_hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    reconstructed_hsv = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2HSV)
    

    original_mask_count = np.count_nonzero(np.maximum(original_hsv[:,:,0]<20, original_hsv[:,:,0]>160))
    reconstructed_mask_count = np.count_nonzero(np.maximum(reconstructed_hsv[:,:,0]<20, reconstructed_hsv[:,:,0]>160))

    
    return float(original_mask_count)/(float(reconstructed_mask_count)+1)#(original_mask_count - reconstructed_mask_count)/original_mask_count

def RedPixelsExtraction(im):
    original_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    original_lightness_mask = original_hsv[:,:,2]<140
    kernel = np.ones((3,3))
    original_lightness_mask = cv2.erode(np.uint8(original_lightness_mask)*255,kernel)>0
    original_mask = np.minimum(np.maximum(original_hsv[:,:,0]<20, original_hsv[:,:,0]>160), original_lightness_mask)
    original_mask_count = np.count_nonzero(original_mask)
    
    return original_mask_count

def PatchesFRedComputation(patches,recons_patches):
    FRed=[]
    for k in np.arange(patches.shape[0]):
        patch=patches[k,:]
        recons=recons_patches[k,:]
        FRed.append(fractionOfRed(patch.astype('uint8'),recons.astype('uint8')))
    
    FRed=np.array(FRed)
    

    return FRed

def AUCOptTh(y_true,y_prob):
    
    if len(y_prob.shape)==1:
        y_prob=np.expand_dims(y_prob,axis=1)
        y_prob=np.concatenate((y_prob,y_prob),axis=1)
    
    roc_auc,fpr,tpr,thresholds=AUCMetrics(y_true,y_prob)    
    
    opt_th=np.argmin((fpr)**2+(1-tpr)**2)
    opt_th=thresholds[opt_th]
    
    return opt_th