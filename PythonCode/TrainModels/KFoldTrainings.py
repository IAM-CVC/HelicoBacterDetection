# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:34:44 2024

@author: debora
"""
import numpy as np
from torch.utils.data import DataLoader
import gc
import torch

from models_init import *

from train_AutoEncoder import train_model as train_model_AE

from datasets import Standard_Dataset
from test_models import eval_modelNN,eval_model_AutoEncoder,eval_model_Encoder
from FC_Networks import Seq_NN
from AEmodels import AutoEncoderCNN


def trainFolds_AE(X,ModelParams,Folds,TRAIN_CONFIG):
    
    inputmodule_paramsEnc=ModelParams['inputmodule_paramsEnc']
    net_paramsEnc=ModelParams['net_paramsEnc']
    inputmodule_paramsDec=ModelParams['inputmodule_paramsDec']
    net_paramsDec=ModelParams['net_paramsDec']
 #   X=np.transpose(X,(0,3, 1, 2))
    
    model_weights=[]
    embeddings_Tr=[]
    embeddings_Ts=[]
    recons_Tr=[]
    recons_Ts=[]
    avg_cost_folds=[]
    for kF in np.arange(0,len(Folds)):
        
        ### TRAIN
        # Train Set
        train_idx=Folds[kF][0]
        train_dataset = Standard_Dataset(X[train_idx,:])
        train_dl = DataLoader(train_dataset,batch_size=TRAIN_CONFIG['batch_size'],shuffle=TRAIN_CONFIG['shuffle'])
        
        # Train Model
        model=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                             inputmodule_paramsDec, net_paramsDec).to('cuda')
    #    init_weights_xavier_normal(model)
        
        model, avg_cost = train_model_AE(model,train_dl, TRAIN_CONFIG)
        
        avg_cost_folds.append(avg_cost)
        ### TEST
        # Evaluate Train
        train_dataset = Standard_Dataset(X[train_idx,:])
        train_dl = DataLoader(train_dataset,batch_size=TRAIN_CONFIG['batch_size'],shuffle=False)
        recons_Tr.append(eval_model_AutoEncoder(model, train_dl))
    #    recons_Tr=np.transpose(recons_Tr,(0,2, 3, 1))
        embeddings_Tr.append(eval_model_Encoder(model,train_dl))
       
        
        # Evaluate Test
        test_idx=Folds[kF][1]
        train_dataset = Standard_Dataset(X[test_idx,:])
        train_dl = DataLoader(train_dataset,batch_size=TRAIN_CONFIG['batch_size'],shuffle=False)
        recons_Ts.append(eval_model_AutoEncoder(model, train_dl))
   #     recons_Ts=np.transpose(recons_Ts,(0,2, 3, 1))
        embeddings_Ts.append(eval_model_Encoder(model,train_dl))
       
    
        model_weights.append(model.to('cpu').state_dict())
        
        print('Fold Finished:', kF)
    # Free GPU Memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return embeddings_Tr,embeddings_Ts,recons_Tr,recons_Ts,model_weights,avg_cost_folds
  
