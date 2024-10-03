# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: 
"""
import sys
import os
import pickle

import numpy as np
import pandas as pd
import glob
import cv2 as cv
from matplotlib import pyplot as plt
from ismember import ismember
from random import shuffle
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold,GroupShuffleSplit
import torch

## Own Functions
CodeDir=r'\Python' 
sys.path.append(CodeDir)
libDirs=next(os.walk(CodeDir))[1]
for lib in libDirs:
    sys.path.append(os.path.join(CodeDir,lib))


from KFoldTrainings import trainFolds_AE
from LoadFunctions import *
from AERedMetrics import AUCOptTh,PatchesFRedComputation
from QualityMetrics import AUCMetrics

 
def AEConfigs(Config):
    
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
       

        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    

        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3
ModelParams={}
ModelParams['inputmodule_paramsEnc']=inputmodule_paramsEnc



# 0.1 NETWORK TRAINING PARAMS
DEVICE='cuda'
torch.cuda.set_device(0)
# Optimizer Parameters
lr=0.01 #lr=0.01 before
# Train Param    
TRAIN_CONFIG = {
        
    'transf'        : False, # tranformation data
       
    'batch_size'    : 128,  # batch_size according to GPU memory available
    'test_size'     : 0, # test size, if zero, NO validation set
    
    'n_epochs'      : 120,  # number of epochs
    'dataset_type'      : 'unbalanced', #whether train set should be balanced (unnecessary in case weighted loss is used) 
    'scaler': None, #'patient' for aggregation of 2D slices
    'shuffle': True,
    'lr':lr
           
}
# 0.2 FOLDERS

DataDir=r'\Data'

ExcelMetaData=['HP_WSI-CoordAnnotatedWindows.xlsx']
patch_csv = os.path.join(DataDir,"PatientDiagnosis.csv")
ResDir=r'\Results'


#### 1. LOAD DATA
# 1.1 Patient Diagnosis
df_diag = pd.read_csv(patch_csv)
neg_folders = df_diag[df_diag['DENSITAT'] == 'NEGATIVA']['CODI'].values
pos_folders = df_diag[df_diag['DENSITAT'] == 'ALTA']['CODI'].values

# 1.2 Patches Data
df_meta=pd.DataFrame()
excelfile=ExcelMetaData[0]
df_meta=pd.concat([df_meta,pd.read_excel(os.path.join(DataDir,excelfile))])

#### 2. DATA SPLITING
# 2.0 Annotated set for FRed optimal threshold
df_meta=df_meta.loc[df_meta.Presence!=0]
pos_anno=np.unique(df_meta['Pat_ID'].values[ismember(
    df_meta['Pat_ID'],pos_folders)[0]])
neg_anno=np.unique(df_meta['Pat_ID'].values[ismember(
    df_meta['Pat_ID'],neg_folders)[0]])

PatID_anno=df_meta.Pat_ID.values
casesID_anno=[PatID+'_'+str(WSI) for PatID,WSI in 
         zip(df_meta.Pat_ID.values,df_meta.WSI_ID.values)]
winID_anno=df_meta.Window_ID.values

y_true_anno=df_meta.Presence.values==1
y_true_anno=y_true_anno.astype(int)

# 2.1 AE trainnig set
N_crop_neg=len(pos_anno)-len(neg_anno)
neg_crop=neg_folders[np.nonzero(~ismember(neg_folders,neg_anno)[0])[0]]

N_crop_neg=35
neg_crop=neg_crop[0:N_crop_neg]
neg_AE=np.concatenate((neg_crop,neg_anno))

# 2.1 Diagosis crossvalidation set
neg_Diag=neg_folders[np.nonzero(~ismember(neg_folders,neg_anno)[0])[0]]
pos_Diag=pos_folders[np.nonzero(~ismember(pos_folders,pos_anno)[0])[0]]

folders_Diag=np.concatenate((neg_Diag,pos_Diag))
y_folders_Diag=np.concatenate((np.zeros(len(neg_Diag)),
                               np.zeros(len(pos_Diag))))
shuffle(folders_Diag)

#### 3. lOAD PATCHES
# 3.1 Load AE Patches
CroppedDir=os.path.join(DataDir,'CroppedPatches','SubImage' )
n_images_patient=150
AEpatches,AEpatches_Pat=loadCroppedPatches(neg_AE,CroppedDir,n_images_patient)
       
# 3.2 Load Annotated Patches
files_anno=[os.path.join(case,str(win)+'.png') 
       for case,win in zip(casesID_anno,winID_anno)]

AnnoDir=os.path.join(DataDir,'AnnotatedPatches')
files_anno=[os.path.join(AnnoDir,'SubImage',file) for file in files_anno]

patches_anno=LoadAnnotatedPatches(files_anno,DataDir)
            

 
## 3.4 Patch size curation
AEpatches,PatID_AEpatches=PatchCuration(AEpatches,AEpatches_Pat)
AEpatches=np.stack(AEpatches)
AEpatches=AEpatches.astype('float32')
PatID_AEpatches=np.array(PatID_AEpatches)


patches_anno=np.stack(patches_anno)
patches_anno=patches_anno.astype('float32')


### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split
samp=2
sgkf=GroupShuffleSplit(n_splits=1,test_size=0.001)
kfold_splitAE=list(sgkf.split(AEpatches[0::samp]
                            ,groups=PatID_AEpatches[0::samp]))



###### CONFIG1
Config='1'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
ModelParams['net_paramsEnc']=net_paramsEnc
ModelParams['inputmodule_paramsDec']=inputmodule_paramsDec
ModelParams['net_paramsDec']=net_paramsDec

TRAIN_CONFIG['n_epochs']=750
TRAIN_CONFIG['loss']='MSE'
TRAIN_CONFIG['batch_size']=250
embeddings_Tr1,embeddings_Ts1,recons_Tr1,recons_Ts1,model_weights1,avg_cost1=trainFolds_AE(AEpatches[0::samp],
                                                          ModelParams,kfold_splitAE,TRAIN_CONFIG)

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
torch.save(model_weights1,file_out+'.tns')
fold_res={'ModelParams': ModelParams,'embeddings_Tr': embeddings_Tr1,'embeddings_Ts': embeddings_Ts1,
          'recons_Tr':recons_Tr1,'recons_Ts':recons_Ts1,'TRAIN_CONFIG':TRAIN_CONFIG}

file = open(file_out+'.pkl', 'wb')
pickle.dump(fold_res,file)

###### CONFIG2
Config='2'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
ModelParams['net_paramsEnc']=net_paramsEnc
ModelParams['inputmodule_paramsDec']=inputmodule_paramsDec
ModelParams['net_paramsDec']=net_paramsDec


TRAIN_CONFIG['n_epochs']=550
TRAIN_CONFIG['loss']='MSE'
TRAIN_CONFIG['batch_size']=250
embeddings_Tr2,embeddings_Ts2,recons_Tr2,recons_Ts2,model_weights2,avg_cost2=trainFolds_AE(AEpatches[0::samp],
                                                          ModelParams,kfold_splitAE,TRAIN_CONFIG)

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
torch.save(model_weights2,file_out+'.tns')
fold_res={'ModelParams': ModelParams,'embeddings_Tr': embeddings_Tr2,'embeddings_Ts': embeddings_Ts2,
          'recons_Tr':recons_Tr2,'recons_Ts':recons_Ts2,'TRAIN_CONFIG':TRAIN_CONFIG}

file = open(file_out+'.pkl', 'wb')
pickle.dump(fold_res,file)

###### CONFIG3
Config='3'#Paper Configuration
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
ModelParams['net_paramsEnc']=net_paramsEnc
ModelParams['inputmodule_paramsDec']=inputmodule_paramsDec
ModelParams['net_paramsDec']=net_paramsDec

TRAIN_CONFIG['n_epochs']=550
TRAIN_CONFIG['loss']='MSE'
TRAIN_CONFIG['batch_size']=250
embeddings_Tr3,embeddings_Ts3,recons_Tr3,recons_Ts3,model_weights3,avg_cost3=trainFolds_AE(AEpatches[0::samp],
                                                          ModelParams,kfold_splitAE,TRAIN_CONFIG)

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
torch.save(model_weights3,file_out+'.tns')
fold_res={'ModelParams': ModelParams,'embeddings_Tr': embeddings_Tr3,'embeddings_Ts': embeddings_Ts3,
          'recons_Tr':recons_Tr3,'recons_Ts':recons_Ts3,'TRAIN_CONFIG':TRAIN_CONFIG}

file = open(file_out+'.pkl', 'wb')
pickle.dump(fold_res,file)

###### CONFIG4
Config='3' #La del paper
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
ModelParams['net_paramsEnc']=net_paramsEnc
ModelParams['inputmodule_paramsDec']=inputmodule_paramsDec
ModelParams['net_paramsDec']=net_paramsDec

TRAIN_CONFIG['n_epochs']=550
TRAIN_CONFIG['loss']='SSIM_Pau'
TRAIN_CONFIG['batch_size']=250
embeddings_Tr31,embeddings_Ts31,recons_Tr31,recons_Ts31,model_weights31,avg_cost31=trainFolds_AE(AEpatches[0::samp],
                                                          ModelParams,kfold_splitAE,TRAIN_CONFIG)

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
torch.save(model_weights31,file_out+'.tns')
fold_res={'ModelParams': ModelParams,'embeddings_Tr': embeddings_Tr31,'embeddings_Ts': embeddings_Ts31,
          'recons_Tr':recons_Tr31,'recons_Ts':recons_Ts31,'TRAIN_CONFIG':TRAIN_CONFIG}

file = open(file_out+'.pkl', 'wb')
pickle.dump(fold_res,file)


#### 4. AE RED METRICS THRESHOLD LEARNING

## 4.1 Model Evaluation
from datasets import Standard_Dataset
from test_models import eval_model_AutoEncoder,eval_model_Encoder
from AEmodels import AutoEncoderCNN
from torch.utils.data import DataLoader

train_dataset = Standard_Dataset(patches_anno)
train_dl = DataLoader(train_dataset,batch_size=TRAIN_CONFIG['batch_size'],shuffle=False)

# CONFIG1

Config='1'
TRAIN_CONFIG['loss']='MSE'
net_paramsEnc,inputmodule_paramsDec=AEConfigs(Config)
model1=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec).to('cuda')

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
model_weights1=torch.load(file_out+'.tns')
model1.load_state_dict(model_weights1[0])
recons_anno1=eval_model_AutoEncoder(model1, train_dl)

# CONFIG2
Config='2'
TRAIN_CONFIG['loss']='MSE'
net_paramsEnc,inputmodule_paramsDec=AEConfigs(Config)
model2=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec).to('cuda')

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
model_weights2=torch.load(file_out+'.tns')
model2.load_state_dict(model_weights2[0])
recons_anno2=eval_model_AutoEncoder(model2, train_dl)

# CONFIG3
Config='3'
TRAIN_CONFIG['loss']='MSE'
net_paramsEnc,inputmodule_paramsDec=AEConfigs(Config)
model3=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec).to('cuda')

file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
model_weights3=torch.load(file_out+'.tns')
model3.load_state_dict(model_weights3[0])
recons_anno3=eval_model_AutoEncoder(model3, train_dl)

TRAIN_CONFIG['loss']='SSIM_Pau'
model31=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec).to('cuda')
file_out=os.path.join(ResDir,TRAIN_CONFIG['loss']+'Config'+Config)
model_weights31=torch.load(file_out+'.tns')
model31.load_state_dict(model_weights31[0])
recons_anno31=eval_model_AutoEncoder(model31, train_dl)

## 4.2 RedMetrics Threshold 
FRed1=PatchesFRedComputation(patches_anno,recons_anno1)
opt_th1=AUCOptTh(y_true_anno,FRed1)

FRed2=PatchesFRedComputation(patches_anno,recons_anno2)
opt_th2=AUCOptTh(y_true_anno,FRed2)

FRed3=PatchesFRedComputation(patches_anno,recons_anno3)
opt_th3=AUCOptTh(y_true_anno,FRed3)

FRed31=PatchesFRedComputation(patches_anno,recons_anno31)
opt_th31=AUCOptTh(y_true_anno,FRed31)

from sklearn import metrics
y_pred_anno=(FRed1>opt_th1).astype(int)
cm1 = metrics.confusion_matrix(y_true_anno, y_pred_anno, normalize='true')

y_pred_anno=(FRed2>opt_th2).astype(int)
cm2 = metrics.confusion_matrix(y_true_anno, y_pred_anno, normalize='true')

y_pred_anno=(FRed3>opt_th3).astype(int)
cm3 = metrics.confusion_matrix(y_true_anno, y_pred_anno, normalize='true')

y_pred_anno=(FRed31>opt_th31).astype(int)
cm31 = metrics.confusion_matrix(y_true_anno, y_pred_anno, normalize='true')


### 5. DIAGNOSIS CROSSVALIDATION
# 5.1 Load Patches 4 CrossValidation of Diagnosis
CroppedDir=os.path.join(DataDir,'CroppedPatches','WSI' )
n_images_patient=150
Diagpatches,Diagpatches_Pat=loadCroppedPatches(folders_Diag,CroppedDir,n_images_patient)
DiagpatchesCured,PatID_Diagpatches=PatchCuration(Diagpatches,Diagpatches_Pat)
Diagpatches=np.stack(DiagpatchesCured)
Diagpatches=Diagpatches.astype('float32')
PatID_Diagpatches=np.array(PatID_Diagpatches)

y_diag_patches=ismember(PatID_Diagpatches,pos_folders)[0].astype(int)
# 5,2 Red Metrics
train_dataset = Standard_Dataset(Diagpatches)
train_dl = DataLoader(train_dataset,batch_size=500,shuffle=False)

recons_diag=eval_model_AutoEncoder(model31, train_dl)
opt_th=opt_th31
RedPixPat=PatchesFRedComputation(Diagpatches,recons_diag)

### 5.2 Diagnostic Power

n_splits=10
sgkf=StratifiedGroupKFold(n_splits=n_splits) #, shuffle=True, random_state=42)
kfold_split=list(sgkf.split(y_diag_patches,
                            y_diag_patches
                            ,groups=PatID_Diagpatches))


auc_diag_Ts=[]
diag_th=[]
diagrec0_Ts=[]
diagrec1_Ts=[]
diagcm_Ts=[]
diagprec0_Ts=[]
diagprec1_Ts=[]
fpr_Ts=[]
tpr_Ts=[]
for kF in np.arange(n_splits):
    
    # Train
    train_idx=kfold_split[kF][0]
    
   # train_idx=(ismember(PatID_Diagpatches,folders_Diag[train_idx])[0])
        
    y_pred_diag_Tr=(RedPixPat[train_idx]>opt_th).astype(int)
    patches_Pat_Tr=PatID_Diagpatches[train_idx]
    d={'y_pred_diag':y_pred_diag_Tr, 'PatID':patches_Pat_Tr}
    df_diag=pd.DataFrame(data=d)
    df_diag=df_diag.groupby('PatID').mean()
    y_diag=ismember(df_diag.index.values,pos_folders)[0].astype(int)
    probDiag=df_diag.values
    opt_th_d=AUCOptTh(y_diag,probDiag[:,0])
    # diag_th.append(opt_th_d)
    
    # y_pred=(probDiag[:,0]>opt_th_d).astype(int)
    # cm = metrics.confusion_matrix(y_diag, y_pred, normalize='true')
    # diagrec0_Tr.append(cm[0,0])
    # diagrec1_Tr.append(cm[1,1])
    
    diag_th.append(opt_th_d)
    
    #Test
    test_idx=kfold_split[kF][1]
 #   test_idx=np.nonzero(1-train_idx)[0]
    
    
    y_pred_diag_Ts=(RedPixPat[test_idx]>opt_th).astype(int)
    patches_Pat_Ts=PatID_Diagpatches[test_idx]
    d={'y_pred_diag':y_pred_diag_Ts, 'PatID':patches_Pat_Ts}
    df_diag=pd.DataFrame(data=d)
    df_diag=df_diag.groupby('PatID').mean()
    y_diag=ismember(df_diag.index.values,pos_folders)[0].astype(int)
    
    probDiag=df_diag.values[:,0]
    y_pred=(probDiag>opt_th_d).astype(int)
    
    probDiag=np.expand_dims(df_diag.values,axis=1)
    roc_auc,fpr,tpr,thresholds=AUCMetrics(y_diag,
                                          np.concatenate((probDiag,probDiag),axis=1))    
       
    auc_diag_Ts.append(roc_auc)
    # fpr_Ts.append(fpr)
    # tpr_Ts.append(tpr)
    fpr_Ts.append(y_diag)
    tpr_Ts.append(df_diag.values[:,0])
    
    cm = metrics.confusion_matrix(y_diag, y_pred, normalize='true')
    diagrec0_Ts.append(cm[0,0])
    diagrec1_Ts.append(cm[1,1])
    
    cm = metrics.confusion_matrix(y_diag, y_pred)
    
    diagprec0_Ts.append(cm[0,0]/np.sum(cm,axis=0)[0])
    diagprec1_Ts.append(cm[1,1]/np.sum(cm,axis=0)[1])
    
    diagcm_Ts.append(cm)

PaperResDir=r'D:\textos\Medical\DigiPatics\Journals\IJCARS\Revision\Results'
df={'fpr':fpr_Ts, 'tpr':tpr_Ts,
    'cm':diagcm_Ts}

df={'y_true':np.concatenate(fpr_Ts), 'y_prob':np.concatenate(tpr_Ts),
    'cm':diagcm_Ts}
np.savez(os.path.join(PaperResDir,'Results_AE_Probas'),df_res=df)


cm=np.zeros((2,2))
for k in np.arange(10):
   
    cm=cm+diagcm_Ts[k]
        
diagprec0_Ts=np.array(diagprec0_Ts)
diagprec1_Ts=np.array(diagprec1_Ts)
diagrec0_Ts=np.array(diagrec0_Ts)
diagrec1_Ts=np.array(diagrec1_Ts)
auc_diag_Ts=np.array(auc_diag_Ts)
F10_Ts=2 * (diagprec0_Ts * diagrec0_Ts) / (diagprec0_Ts + diagrec0_Ts)
F11_Ts=2 * (diagprec1_Ts * diagrec1_Ts) / (diagprec1_Ts + diagrec1_Ts)

print(np.mean(diagrec0_Ts[np.nonzero(diagrec0_Ts)[0]]))
print(np.std(diagrec0_Ts[np.nonzero(diagrec0_Ts)[0]]))
print(np.mean(diagrec1_Ts[np.nonzero(diagrec1_Ts)[0]]))
print(np.std(diagrec1_Ts[np.nonzero(diagrec1_Ts)[0]]))

print(np.mean(diagprec0_Ts[np.nonzero(diagprec0_Ts)[0]]))
print(np.std(diagprec0_Ts[np.nonzero(diagprec0_Ts)[0]]))
print(np.mean(diagprec1_Ts[np.nonzero(diagprec1_Ts)[0]]))
print(np.std(diagprec1_Ts[np.nonzero(diagprec1_Ts)[0]]))

print(np.mean(F10_Ts[np.nonzero(F10_Ts)[0]]))
print(np.std(F10_Ts[np.nonzero(F10_Ts)[0]]))
print(np.mean(F11_Ts[np.nonzero(F11_Ts)[0]]))
print(np.std(F11_Ts[np.nonzero(F11_Ts)[0]]))