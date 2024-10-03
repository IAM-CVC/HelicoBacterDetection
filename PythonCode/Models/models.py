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
from collections import OrderedDict
from numpy.matlib import repmat

from models_init import *
from FC_Networks import OneShotTriplet_NN


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
        self, num_input_features: int, n_neurons: int, kernel_sze:int =3, drop_rate: float=0,
        dim=2, Relu=True
         ) -> None:
        super().__init__()
        
        if dim==2:
            norm1 = nn.BatchNorm2d(n_neurons)
            conv1 = nn.Conv2d(num_input_features, n_neurons, kernel_size=kernel_sze,  
                                   stride=(int((kernel_sze-1)/2)), padding=(int((kernel_sze-1)/2)))
        elif dim==3:
            norm1 = nn.BatchNorm3d(n_neurons)
            conv1 = nn.Conv3d(num_input_features, n_neurons,kernel_size=kernel_sze, 
                                   stride=(int((kernel_sze-1)/2)),padding=(int((kernel_sze-1)/2))
        
                                  )
        relu1 = nn.ReLU(inplace=True)

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
        self, num_input_features: int, n_neurons: int, kernel_sze:int =3, drop_rate: float=0,
        dim=2, Relu=False
         ) -> None:
        super().__init__()
        
        if dim==2:
            norm1 = nn.BatchNorm2d(n_neurons)
            conv1 = nn.ConvTranspose2d(num_input_features, n_neurons, kernel_size=kernel_sze,  
                                   stride=(int((kernel_sze-1)/2)), padding=(int((kernel_sze-1)/2)))
        elif dim==3:
            norm1 = nn.BatchNorm3d(n_neurons)
            conv1 = nn.ConvTranspose3d(num_input_features, n_neurons,kernel_size=kernel_sze, 
                                   stride=(int((kernel_sze-1)/2)),padding=(int((kernel_sze-1)/2))
                                   )
        
        relu1 = nn.ReLU(inplace=True)

        drop=nn.Dropout(drop_rate)
        if Relu:
            self.cnn_layer=nn.Sequential(conv1,norm1,relu1,drop)
        else:
            self.cnn_layer=nn.Sequential(conv1,norm1,drop)
        init_weights_xavier_normal(self)
        
    def forward(self, x: Tensor):
         
        return(self.cnn_layer(x))
    
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
        dim=2,
        decoder=False,
        Relu=True
    
    ) -> None:
        super().__init__()
        
        num_layers=len(block_config)
        self.num_input_channels=num_input_channels
        for i in range(num_layers):
            if decoder==True:
                layer = _UnCNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    drop_rate=drop_rate, dim=dim
                    
                )
            else:
                layer = _CNNLayer(
                    num_input_channels,
                    n_neurons=block_config[i],
                    drop_rate=drop_rate, dim=dim,
                    Relu=Relu
                    
                )
            self.add_module("cnnlayer%d" % (i + 1), layer)
            num_input_channels=block_config[i]

    def forward(self, x: Tensor) -> Tensor:
        
        for name, layer in self.items():
            x = layer(x)
            
            
        return x


# Models
class AutoEncoderCNN(nn.Module):
    r"""AutoEncoderCNN model class
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
           # outchannels_encoder.append(num_input_channels)
            
        # Decoder
        self.decoder=nn.Sequential(          
            )
        
        for i in np.arange(n_blocks)[::-1]:
            block = _CNNBlock(
                num_input_channels=num_input_channels,
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

        x=self.encoder(x)
        x=self.decoder(x)
        x=F.upsample(x,size=input_sze[2::],mode=self.upPoolMode)
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
    
class OneShotCNNNet(nn.Module):
    r"""OneShotCNNNet-BC model class
    `".
    """

    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        self.dim=net_params['dim']
        
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs']
        n_blocks=len(block_configs)
        
        if 'prct' not in list(outmodule_params.keys()):
            Nprct=1
            self.prct=None
        else:
            Nprct=len(outmodule_params['prct'])
            self.prct=outmodule_params['prct']
            
        self.features=nn.Sequential(          
            )
        
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                dim=self.dim
            )
            self.features.add_module("cnnblock%d" % (i + 1), block)
            if self.dim==2:
                self.features.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif self.dim==3:
                self.features.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            num_input_channels=block_configs[i][-1]
            
        if self.dim==2:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        
        elif self.dim==3:
            self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 5, 5))
            
        classifier_inputmodule_params={}
        classifier_inputmodule_params['n_inputs']=np.prod(
            self.avgpool.output_size)*block_configs[-1][-1]*Nprct

        self.classifier=LinearBlock(classifier_inputmodule_params, outmodule_params)
    
    def feature_extraction(self,x):
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return x
    
    
    def forward(self, x: Tensor) -> Tensor:
        
        # Feature extraction
        x = self.feature_extraction(x)
        if self.prct is not None:
            x_nod=torch.quantile(x,q=self.prct[0],dim=0)
            for prct in self.prct[1::]:
                x_nod= torch.concatenate((x_nod,torch.quantile(x,q=prct,dim=0))
                                         )
            # x= torch.concatenate((torch.quantile(x,q=0.25,dim=0),
            #                         torch.quantile(x,q=0.5,dim=0),
            #                         torch.quantile(x,q=0.75,dim=0)))
            x_nod=torch.tile(x_nod,(1,1))
        else:
            x_nod=x
        # OneShot output
        out=self.classifier(x_nod) 
        
        return out



