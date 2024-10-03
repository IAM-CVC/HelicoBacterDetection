# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:30:28 2020

@author: jyauri
"""
import pandas as pd
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import gc


def eval_modelNN(model, test_dataloader):
    
   

    total_test_batch = len(test_dataloader)
    DEVICE= list(model.parameters())[1].device.type
    
    output=[]
    
    model.eval()
    with torch.no_grad():
        iter_test_dataset = iter(test_dataloader)
        for j in range(total_test_batch):
            seqs, targets = next(iter_test_dataset)
            seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
            outs = model(seqs)
            outs=outs.cpu().detach().numpy()
            
         
            output.append(outs)
           
    del outs
    gc.collect()
    torch.cuda.empty_cache()
          
    output = np.concatenate(output,axis=0)

    
    return output

def eval_model(model, test_dataloader):
    
   

    total_test_batch = len(test_dataloader)
    
    
    output=[]
    
    model.eval()
    with torch.no_grad():
        iter_test_dataset = iter(test_dataloader)
        for j in range(total_test_batch):
            seqs, targets = next(iter_test_dataset)
            seqs, targets = seqs.cuda(), targets.cuda()
            outs = (model(seqs.permute(0, 3, 1, 2)))
            outs=outs.cpu().detach().numpy()
            
         
            output.append(outs)
           
    del outs
    gc.collect()
    torch.cuda.empty_cache()
          
    output = np.concatenate(output,axis=0)

    
    return output

def eval_model_Encoder(model, test_dataloader,szepool=7):
    
  

        total_train_batch = len(test_dataloader)

        outputs=[]
        outputs_resized=[]
        model.eval()
        with torch.no_grad():
            iter_train_dataset = iter(test_dataloader)
            for k in range(total_train_batch):
                # batch 
                seqs = next(iter_train_dataset)
                if list(model.parameters())[1].device.type=='cuda':
                   
                    seqs = seqs.cuda()
              
                # model evaluation
                seqs=seqs.permute(0, 3, 1, 2)

                outs=model.encoder(seqs)  
                outs_resized=nn.AdaptiveAvgPool2d(szepool)(outs)

                outs=torch.squeeze(outs).cpu().detach().numpy()
                outs_resized=torch.squeeze(outs_resized).cpu().detach().numpy()
                outputs.append( outs)
                outputs_resized.append(outs_resized)
                
            outputs=np.concatenate(outputs,axis=0)
            outputs_resized=np.concatenate(outputs_resized,axis=0)
    
        return outputs,outputs_resized
                
def eval_model_AutoEncoder(model, test_dataloader):
    
  

        total_train_batch = len(test_dataloader)

        outputs=[]
        
        model.eval()
        with torch.no_grad():
            iter_train_dataset = iter(test_dataloader)
            for k in range(total_train_batch):
                # batch 
                seqs = next(iter_train_dataset)
                if list(model.parameters())[1].device.type=='cuda':
                   
                    seqs = seqs.cuda()
              
                # model evaluation
                outs=model(seqs.permute(0, 3, 1, 2))  
                
                            
          
                outs=torch.squeeze(outs.permute(0,2,3,1)).cpu().detach().numpy()
                outputs.append( outs)
            
            
        gc.collect()
        torch.cuda.empty_cache()    
        
        return np.concatenate(outputs,axis=0)
    
    
def eval_model_1Shot(model, test_dataloader):
    
   

        total_train_batch = len(test_dataloader)

        outputs=[]
        targets=[]
        PatIDs=[]
        
        model.eval()
        with torch.no_grad():
            iter_train_dataset = iter(test_dataloader)
            for k in range(total_train_batch):
                
                # batch 
                seqs, target = next(iter_train_dataset)
                if list(model.parameters())[1].device.type=='cuda':
                   
                    seqs = seqs.cuda()
              
                # model evaluation
                if model.dim==2:
                    seqs=seqs.transpose(0,1)
                else:
                    seqs=torch.tile(seqs,(1,1,1,1,1))
                    
                target=target[0]
                
                outs=model(seqs)   
                
                # out_nod= torch.concatenate((torch.quantile(out_nod,q=0.25,dim=0),
                #                         torch.quantile(out_nod,q=0.5,dim=0),
                #                         torch.quantile(out_nod,q=0.75,dim=0)))
                
                # out=model.feature_extraction(seqs)
                # out_nod=torch.quantile(out,q=model.prct[0],dim=0)
                # for prct in model.prct[1::]:
                #     out_nod= torch.concatenate((out_nod,
                #                                 torch.quantile(out,q=prct,dim=0))
                #                               )
                
                outs=outs.cpu().detach().numpy()
                outputs.append( outs)
                targets.append(target[0].repeat(seqs.shape[0]).numpy())
                PatIDs.append(target[1].repeat(seqs.shape[0]).numpy())
            
           # outputs=torch.stack(outputs)
            outputs=np.concatenate(outputs,axis=0)
            targets=np.concatenate(targets,axis=0)
            PatIDs=np.concatenate(PatIDs,axis=0)
            
        return outputs,targets,PatIDs