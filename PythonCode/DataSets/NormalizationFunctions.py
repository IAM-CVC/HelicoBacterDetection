# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:27:01 2024

@author: debora
"""

import numpy as np

### Normalization Functions    
def Normalize(patches):

    patchesN=patches.astype('float32').copy()
    mu=[np.mean(patches[:,:,:,0].flatten()),
        np.mean(patches[:,:,:,1].flatten()),np.mean(patches[:,:,:,2].flatten())]
    std=[np.std(patches[:,:,:,0].flatten()),
         np.std(patches[:,:,:,1].flatten()),np.std(patches[:,:,:,2].flatten())]
    for kch in np.arange(3):
        patchesN[:,:,:,kch]=(patches[:,:,:,kch]-mu[kch])/std[kch]

    return patchesN,mu,std   

def NormalizeMnMx(patches):

    patchesN=patches.astype('float32').copy()
    Mn=[np.min(patches[:,:,:,0].flatten()),
        np.min(patches[:,:,:,1].flatten()),np.min(patches[:,:,:,2].flatten())]
    Mx=[np.max(patches[:,:,:,0].flatten()),
         np.max(patches[:,:,:,1].flatten()),np.max(patches[:,:,:,2].flatten())]
    for kch in np.arange(3):
        patchesN[:,:,:,kch]=(patches[:,:,:,kch]-Mn[kch])/(Mx[kch]-Mn[kch])

    return patchesN,Mn,Mx   

from scipy.ndimage import gaussian_filter
def GaussSmooth(patches,sigma=0.5):
    
    patchesN=patches.astype('float32').copy()
    
    for kch in np.arange(patchesN.shape[0]):
        patchesN[kch,:]=gaussian_filter(patchesN[kch,:],sigma=sigma)
    
    return patchesN