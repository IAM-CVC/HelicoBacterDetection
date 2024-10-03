# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:24:08 2024

@author: debora
"""
import glob
from random import shuffle
from skimage import io
import os
import numpy as np


### I/O Functions
def loadCroppedPatches(PatList,cropped_path,n_images_patient):

    pos_files = []
    PatList_read=[]
    for folder in PatList:
       # random.seed = 0
        if len(glob.glob(os.path.join(cropped_path,folder+'*')))>0:
            Patfolder=glob.glob(os.path.join(cropped_path,folder+'*'))[0]
            patient_files =glob.glob(os.path.join(Patfolder,"*.png"))
            shuffle(patient_files)
            pos_files += patient_files[:n_images_patient]
            PatList_read.append(folder)
            
    
    patches = [io.imread(patch)[:,:,:3] for patch in pos_files]
    
    return patches, np.repeat(PatList_read,n_images_patient)

def LoadAnnotatedPatches(files,DataDir):
    
#    files=[os.path.join(case,str(win)+'.png') for case,win in zip(casesID_anno,winID)]
    patches=[]
    # winID_patches=[]
    # casesID_patches=[]
    # files_patches=[]
    # y_true_patches=[]
    for k in np.arange(len(files)):
        
        file=files[k]
        # file=os.path.join(DataDir,'SubImage',file)
        if os.path.isfile(file):
            im = io.imread(os.path.join(DataDir,'SubImage',file))
           
            patches.append(im[:,:,0:3])
            # winID_patches.append(winID[k])
            # casesID_patches.append(casesID_anno[k])
            # files_patches.append(files[k])
            # y_true_patches.append(y_true[k])
    return patches

def PatchCuration(AEpatches,patches_Pat):
    sh=[pt.shape for pt in AEpatches]
    sh=np.array(sh)
    idx_out=np.nonzero(np.min(sh[:,0:2],axis=1)!=256)[0]
    AEpatchesCured=[]
    patches_PatCured=[]
    for k in np.arange(len(AEpatches)): 
        if k not in idx_out:
            AEpatchesCured.append(AEpatches[k])
            patches_PatCured.append(patches_Pat[k])
    
    return AEpatchesCured, patches_PatCured