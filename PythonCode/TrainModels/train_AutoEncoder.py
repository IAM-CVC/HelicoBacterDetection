# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:00:28 2021

@author: debora
"""
from torch import optim as optim
from torch import cuda as cuda
import gc
import numpy as np
import copy

from torch import nn 
from SSIM_loss import SSIM 
from SSIMPau_loss import NegSSIM as SSIM_Pau
from EarlyStopping import * 

def train_model(model,train_dl,TRAIN_CONFIG):
    
    DEVICE= list(model.parameters())[1].device.type
    
    if TRAIN_CONFIG['loss']=='MSE':
        criterion =nn.MSELoss(reduction='mean')
    elif TRAIN_CONFIG['loss']=='SSIM_Pau':
        iter_train_dataset = iter(train_dl)
        MxRang=0
        for seqs in iter_train_dataset:
            MxRang=max(MxRang,seqs.max().numpy())
            
        criterion = SSIM_Pau(max_val=MxRang)
    else:
  # criterion = torch.nn.L1Loss()
        
        iter_train_dataset = iter(train_dl)
        MxRang=0
        for seqs in iter_train_dataset:
            MxRang=max(MxRang,seqs.max().numpy())
        criterion = SSIM(max_val=MxRang)
    
    optimizer = optim.Adam(model.parameters(),lr = TRAIN_CONFIG['lr'])
  #  scaler = torch.cuda.amp.GradScaler()
    
    early_stopping = EarlyStopping(warm_up=60, patience=10)
    losses_train = []
    
    total_train_batch = len(train_dl) 
    model.train()          
    for epoch in np.arange(TRAIN_CONFIG['n_epochs']):
        iter_train_dataset = iter(train_dl)
        loss_acum = []
        for step in range(total_train_batch):
            # batch of nodes 
            seqs = next(iter_train_dataset)
            seqs = seqs.to(DEVICE)
            
            seqs=seqs.permute(0, 3, 1, 2)
            recon_seqs=model(seqs)
            loss = criterion(seqs, recon_seqs)
            loss.backward()
            optimizer.step()
            
            loss_acum.append(loss.cpu().detach().numpy())
            
        losses_train.append(np.mean(loss_acum))
        print('Epoch: {}/{} — Loss: {:.4f}\n'.format(epoch+1, TRAIN_CONFIG['n_epochs'], np.mean(loss_acum)))
        if early_stopping(epoch+1, losses_train[epoch], copy.deepcopy(model)):   
            print('Early Stop at' + str(epoch))
            break
        
    gc.collect()
    cuda.empty_cache()
    
    return model,losses_train
