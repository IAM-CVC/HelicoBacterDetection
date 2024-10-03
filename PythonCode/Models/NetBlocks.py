# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:18:44 2023

@author: debora
"""

import math
import numpy as np
import itertools

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from numpy.matlib import repmat

from models_init import *



### Linear FC Blocks
def linear_block(n_inputs_loc, hidden_loc, 
                 activ_config=None,batch_config=None,p_drop_loc=0.1): 
    
    # Dictionary defining Block Architecture
    BlockArchitecture=[]
   
    hidden_loc.insert(0,n_inputs_loc)
  
    if activ_config==None:
        activ_config=repmat('no_activ',len(hidden_loc),1)
    if batch_config==None:
        batch_config=repmat('no_batch',len(hidden_loc),1)
    #Block Layers List
    for i in np.arange(len(hidden_loc)-1):
        BlockArchitecture.append(('linear'+str(i+1),
                                  nn.Linear(hidden_loc[i], hidden_loc[i+1])))
        
        if(activ_config[i]=='relu'):
            BlockArchitecture.append(('relu'+str(i+1),nn.ReLU(inplace=True)))
           
        elif(activ_config[i]=='tanh'):
            BlockArchitecture.append(('tanh'+str(i+1),nn.Tanh()))
        elif(activ_config[i]=='relu6'):
             BlockArchitecture.append(('relu6'+str(i+1),nn.ReLU6(inplace=True)))
             
        if(batch_config[i]=='batch'):
            BlockArchitecture.append(('batch'+str(i+1),nn.BatchNorm1d( hidden_loc[i+1])))
         
        BlockArchitecture.append(('drop'+str(i+1),nn.Dropout(p_drop_loc)))  
    linear_block_loc = nn.Sequential(
        OrderedDict(BlockArchitecture)
        )
    return linear_block_loc


class LinearBlock(nn.Module):
    """
    MultiLayer Perceptron: 
    Netwotk with n_hidden layers with architecture linear+drop+relu+batch
     Constructor Parameters:
           n_inputs: dimensionality of input features (n_channels * n_features , by default) 
                     n_channels (=14), number of sensors or images for each case
                     n_features(=40), number of features for each n_channels
           n_classes: number of output classes (=3, by default)
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)

    """
    
    def __init__(self, inputmodule_params,net_params):
        super().__init__()

        
       
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
    
            self.activ_config=None
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config=None
        else:
            self.batch_config=net_params['batch_config']
             
              
        
        self.linear_block0= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)

       
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
              
      
        return self.linear_block0(x)

### Convolutional 
class _CNNLayer(nn.Module):
    def __init__(
        self, num_input_features: int, n_neurons: int, kernel_sze:int =3, 
        stride:int=1,
        drop_rate: float=0,
        Relu=True
         ) -> None:
        super().__init__()
        

        norm1 = nn.BatchNorm2d(n_neurons)
        conv1 = nn.Conv2d(num_input_features, n_neurons, kernel_size=kernel_sze,  
                               stride=stride, padding=(int((kernel_sze-1)/2)))

      #  relu1 = nn.ReLU(inplace=True)
        relu1= nn.LeakyReLU(inplace=True)

        drop=nn.Dropout(drop_rate)
        if Relu:
            self.cnn_layer=nn.Sequential(conv1,norm1,relu1,drop)
        else:
            self.cnn_layer=nn.Sequential(conv1,norm1,drop)
        init_weights_xavier_normal(self)
        
    def forward(self, x: Tensor):
         
        return(self.cnn_layer(x))

class _UnCNNLayer(nn.Module):
    def __init__(
        self, num_input_features: int, n_neurons: int, kernel_sze:int =3, 
        stride:int=2,
        drop_rate: float=0, 
        Relu=True
         ) -> None:
        super().__init__()
        
        self.stride=stride
        norm1 = nn.BatchNorm2d(n_neurons)
        conv1 = nn.ConvTranspose2d(num_input_features, n_neurons, kernel_size=kernel_sze,  
                               stride=stride, padding=(int((kernel_sze-1)/2)))

        
     #   relu1 = nn.ReLU(inplace=True)
        relu1 = nn.LeakyReLU(inplace=True)

        drop=nn.Dropout(drop_rate)
        if Relu:
            self.cnn_layer=nn.Sequential(conv1,norm1,relu1,drop)
        else:
            self.cnn_layer=nn.Sequential(conv1,norm1,drop)
        init_weights_xavier_normal(self)
        
    def forward(self, x: Tensor):
        
        if  self.stride>1:
            sze_enc=x.shape[-1]
            x=self.cnn_layer[0](x,output_size=(sze_enc*2,sze_enc*2))
            for k in np.arange(1,len(self.cnn_layer)):
                x=self.cnn_layer[k](x)
        else:
            x=self.cnn_layer(x)
            
        return(x)
    
    # def forward(self, x1, x2):
    #     x1 = self.cnn_layer(x1)
    #     # input is CHW
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]

    #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                     diffY // 2, diffY - diffY // 2])
    #     # if you have padding issues, see
    #     # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    #     # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    #     x = torch.cat([x2, x1], dim=1)
    #     return self.conv(x)
    
class _CNNBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_input_channels: int=1,
        drop_rate=0,
        block_config = (64,128),
        stride=None,
        decoder=False,
        Relu=True
    
    ) -> None:
        super().__init__()
        
        num_layers=len(block_config)
        self.num_input_channels=num_input_channels
        
        if stride is None:
            stride=np.ones(num_layers)
            
        for i in range(num_layers):
            if decoder==True:
                layer = _UnCNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    stride=stride[i],
                    drop_rate=drop_rate
                    
                )
            else:
                layer = _CNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    stride=stride[i],
                    drop_rate=drop_rate, 
                    Relu=Relu
                    
                )
            self.add_module("cnnlayer%d" % (i + 1), layer)
            num_input_channels=block_config[i]

    def forward(self, x: Tensor) -> Tensor:
        
        for name, layer in self.items():
            x = layer(x)
            
            
        return x

