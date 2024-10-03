# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:18:03 2022

@author: debora
"""
import math
import numpy as np
import itertools

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy.matlib import repmat

from models_init import *
from NetBlocks import _CNNBlock,_UnCNNLayer



#  BackBone Modules
class Encoder(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    """

    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        

            
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs'].copy()
        n_blocks=len(block_configs)
        if 'stride' in net_params.keys():
            stride=net_params['stride']
        else:
            stride=[]
            for i in np.arange(len(block_configs)):
                stride.append(list(np.ones(len(block_configs[i])-1,dtype=int))+[2])
                
        # Encoder
        self.encoder=nn.Sequential(          
            )
        outchannels_encoder=[]
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i],
                stride= stride[i]               
                
            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            
            if stride==1:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

            num_input_channels=block_configs[i][-1] 
           # outchannels_encoder.append(num_input_channels)
           
          
               
    def forward(self, x: Tensor) -> Tensor:
        
        x=self.encoder(x)

        return x

class Decoder(nn.Module):
    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
   
        num_input_channels=inputmodule_params['num_input_channels']
        
        self.upPoolMode='bilinear'

            
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs'].copy()
        self.n_blocks=len(block_configs)
        
        if 'stride' in net_params.keys():
            stride=net_params['stride']
        else:
            stride=[]
            for i in np.arange(len(block_configs)):
                stride.append(list(np.ones(len(block_configs[i])-1,dtype=int))+[2])
                

        # Decoder
        self.decoder=nn.Sequential(          
            )
        
        for i0 in np.arange(self.n_blocks)[::-1]:
            i=self.n_blocks-(i0+1)
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                stride=stride[i],
                decoder=True
            )
            
            # if stride==1:
            #     self.decoder.add_module("uppool%d" % (i + 1), 
            #                               nn.Upsample(scale_factor=2, 
            #                                           mode=self.upPoolMode, align_corners=True))
            
            self.decoder.add_module("cnnblock%d" % (i0+1), block)
      

            num_input_channels=block_configs[i][-1]
        
        
        self.decoder[-1][list(self.decoder[-1].keys())[-1]].cnn_layer[2]=nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape

     #   for i in np.arange(n_blocks)[::-1]:
            
        x=self.decoder(x)
    
        return x

##### Generative Models
class AutoEncoderCNN(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    """

    def __init__(self, inputmodule_paramsEnc,net_paramsEnc,inputmodule_paramsDec,net_paramsDec):
        super().__init__()
        
        
 
        self.inputmodule_paramsEnc=inputmodule_paramsEnc
        self.inputmodule_paramsDec=inputmodule_paramsDec
        self.net_paramsEnc=net_paramsEnc
        self.net_paramsDec=net_paramsDec
        # Encoder
        
        self.encoder=Encoder(inputmodule_paramsEnc,net_paramsEnc)
     
        # Decoder
        self.decoder=Decoder(inputmodule_paramsDec,net_paramsDec)
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape

        x=self.encoder(x)
        x=self.decoder(x)
       # x=F.upsample(x,size=input_sze[2::],mode=self.upPoolMode)
        
        return x

class UnetCNN(nn.Module):
    r"""OneShotCNNNet-BC model class
    `".
    """

    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        self.dim=net_params['dim']
        self.upPoolMode='bilinear'
        if self.dim==3:
            self.upPoolMode='trilinear'
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs']
        n_blocks=len(block_configs)
        
        self.prct=None
        # Encoder
        self.encoder=nn.Sequential(          
            )
        outchannels_encoder=[]
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                dim=self.dim
                
            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            if self.dim==2:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif self.dim==3:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            num_input_channels=block_configs[i][-1] 
            outchannels_encoder.append(num_input_channels)
            
        # Decoder
        self.decoder=nn.Sequential(          
            )
        
        for i in np.arange(n_blocks)[::-1]:
            block = _CNNBlock(
                num_input_channels=num_input_channels+outchannels_encoder[i],
                drop_rate=drop_rate,
                block_config=block_configs[i][::-1], 
                dim=self.dim,
                decoder=True
            )
            self.decoder.add_module("uppool%d" % (i + 1), 
                                      nn.Upsample(scale_factor=2, 
                                                  mode=self.upPoolMode, align_corners=True))

            self.decoder.add_module("cnnblock%d" % (i + 1), block)
      

            num_input_channels=block_configs[i][0]
        
        block =  _UnCNNLayer(
            num_input_channels,
            n_neurons=inputmodule_params['num_input_channels'],
            drop_rate=drop_rate, dim=self.dim
            
         )
        self.decoder.add_module("cnnblock%d" % (i), block)
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape
        n_blocks=int(len(list(self.encoder.named_children()))/2)
        #encoder
        x_curr=[]
        for i in np.arange(n_blocks):
            x=self.encoder.get_submodule("cnnblock%d" % (i + 1))(x)
            x_curr.append(x)
            x=self.encoder.get_submodule("mxpool%d" % (i + 1))(x)
                   
       
        #decoder with skip connections
        for i in np.arange(n_blocks)[::-1]:
            x = self.decoder.get_submodule("uppool%d" % (i + 1))(x) 
            # diffY = x_curr[i].size()[-2] - x.size()[-2]
            # diffX = x_curr[i].size()[-1] - x.size()[-1]

            # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
            #                 diffY // 2, diffY - diffY // 2])
            x=F.upsample(x,size=x_curr[i].size()[2::],mode=self.upPoolMode)
            x=torch.cat([x_curr[i], x], dim=1)
            x=self.decoder.get_submodule("cnnblock%d" % (i + 1))(x)
            
        x=self.decoder.get_submodule("cnnblock%d" % (0))(x)    

        return x    
    



