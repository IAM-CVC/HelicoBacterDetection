import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import numpy as np
from numpy.matlib import repmat

from models_init import *
from weights_init import weights_init

from NetBlocks import *



# =============================================================================
#
# =============================================================================

##      FULLY CONNECTED MODELS
# 1. MultiLayer Perceptron
#          
class Seq_NN(nn.Module):
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
    
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        
        self.model_type = 'Simple'
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
      #  if net_params['relu_config'] is None:
            self.activ_config=None
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config=None
        else:
            self.batch_config=net_params['batch_config']
             
        self.n_classes= outmodule_params['n_classes']
       # self.activation=outmodule_params['activation']
        
        
        self.linear_block= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)

        self.fc_out=nn.Sequential(nn.Linear(self.hidden[-1], self.n_classes))
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)

    def get_embedding(self, x):
        return self.linear_block(x)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        x = self.fc_out(x)
        return x

# 2. One-Shot Network


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # projector
        sizes = [args['input_dim']] + args['projector']
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def forward(self, x):
        z1 = self.projector(x[:,0,:])
        z2 = self.projector(x[:,1,:])

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        

        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = self.off_diagonal(c).pow_(2).sum()
        # loss = on_diag + self.args.lambd * off_diag
        
        return c,z1,z2
    
    
class OneShot_NN(nn.Module):
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
    
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        
        self.model_type = 'Simple'
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
      #  if net_params['relu_config'] is None:
            self.activ_config='relu'
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config='batch'
        else:
            self.batch_config=net_params['batch_config']
             
        self.n_classes= outmodule_params['n_classes']
       # self.activation=outmodule_params['activation']
        
        
        self.linear_block1= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)
        self.linear_block2= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)
        self.fc_out=nn.Sequential(nn.Linear(self.hidden[-1], self.n_classes),nn.Sigmoid())
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)

    def get_embedding(self, x):
        return self.linear_block(x)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x1 = self.linear_block1(x[:,0,:])
        x2 = self.linear_block2(x[:,1,:])
        x = self.fc_out(abs(x1-x2))
        return x

class OneShotTriplet_NN(nn.Module):
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

        
        self.model_type = 'Triplet'
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
      #  if net_params['relu_config'] is None:
            self.activ_config='relu'
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config='batch'
        else:
            self.batch_config=net_params['batch_config']
             
              
        
        self.linear_block0= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)

       
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)

    def get_embedding(self, x):
        return self.linear_block0(x[:,0,:])

    def forward(self, x):
              

        x1 = self.linear_block0(x[:,0,:])
        x2 = self.linear_block0(x[:,1,:])
        x3 = self.linear_block0(x[:,2,:])
       
        return x1,x2,x3
    

class OneShotTripletSiamese_NN(nn.Module):
    """

    """
    
    def __init__(self, inputmodule_params,net_params):
        super().__init__()

        
        self.model_type = 'Triplet'
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
      #  if net_params['relu_config'] is None:
            self.activ_config=None
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config=None
        else:
            self.batch_config=net_params['batch_config']
             
       
       # self.activation=outmodule_params['activation']
        
        
        self.linear_block0= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)
        self.linear_block1= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)
       
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)

    def get_embedding(self, x):
        return self.linear_block(x)

   
    def forward(self, x,y):
          
        idx0=np.nonzero(y==0)[:,0]
        x1 = self.linear_block0(x[idx0,0,:])
        x2 = self.linear_block0(x[idx0,1,:])
        x3 = self.linear_block1(x[idx0,2,:])
        
        idx0=np.nonzero(y==1)[:,0]
        x1 = torch.cat((x1,self.linear_block1(x[idx0,0,:])))
        x2 = torch.cat((x2,self.linear_block1(x[idx0,1,:])))
        x3 = torch.cat((x3,self.linear_block0(x[idx0,2,:])))
            
        return x1,x2,x3
    
class OneShot_NN_Guille(nn.Module):
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
    
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        
        self.model_type = 'Simple'
        ### Input Parameters
        self.n_inputs = inputmodule_params['n_inputs']

       
        self.hidden=net_params['hidden']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=len(self.hidden)
        if 'activ_config' not in list(net_params.keys()):
      #  if net_params['relu_config'] is None:
            self.activ_config=None
        else:
             self.activ_config=net_params['activ_config']
        
        if 'batch_config' not in list(net_params.keys()):
            self.batch_config=None
        else:
            self.batch_config=net_params['batch_config']
             
        self.n_classes= outmodule_params['n_classes']
       # self.activation=outmodule_params['activation']
        
        
        self.linear_block= linear_block(self.n_inputs, self.hidden.copy(), 
                                                 activ_config=self.activ_config, 
                                                 batch_config=self.batch_config,
                                                 p_drop_loc=self.dropout)
        self.fc_out=nn.Sequential(nn.Linear(self.hidden[-1], self.n_classes),nn.Sigmoid())
        
      #  self.fc_out=nn.Identity()
        # weight init
        init_weights_xavier_normal(self)



    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x1 = self.linear_block(x[0])
        x2 = self.linear_block(x[1])
        x = self.fc_out(abs(x1-x2))
        return x

# 2. MultiLayer Perceptron Ensemble combining Features
#  
class Seq_NN_Ensemble_By_Features(nn.Module):
    """
    MultiLayer Perceptron Ensemble that combines the output 
    features of n_channel MLP networks with architecture Seq_NN
    Constructor Parameters:
           n_channels (=14, default): number of sensors or images for each case
           n_features(=40, default): number of features for each n_channels
           n_classes(=3, by default): number of output classes 
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)

    """

    def __init__(self, n_channels=14, n_features=40, n_classes=3, batch_config=None,
                 hidden=[128],p_drop=0.1,relu_config=None):
        super().__init__()

        
        self.hidden=hidden.copy()
        self.n_nets = n_channels
        self.n_features=n_features
        self.n_classes=n_classes
        self.sensors = np.arange(n_channels)

         # Generate n_nets classification networks (self.model_arr), 
         # one for each input sensor
     
        n_output=self.hidden.pop()
   
        self.model_arr = nn.ModuleList([ ])
        for i in range(self.n_nets):
            self.model_arr.append(linear_block(
                n_features, self.hidden.copy(), n_output,
                relu_config=relu_config, 
                batch_config=batch_config,p_drop_loc=p_drop))
        self.hidden.append(n_output)    
        
        # Agregate self.model_arr output to be the input of the 
        # network that fuses all sensors
        self.fc_out = nn.Sequential(
          
            nn.Linear(self.n_nets * n_output, n_classes),
            nn.Sigmoid(),
            )
        
        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
       
        out = []
        for i in range(self.n_nets):
            data = x[:,i,:]
            data = self.model_arr[i](data) # the best at 200 epochs
            out.append(data)

        out = torch.stack(out, dim=1) # [N, models, outs]
        out = out.view(out.size(0), -1) # [N, models * outs]
        out = self.fc_out(out)
        return out

# 3. MultiLayer Perceptron Ensemble combining Networks Probabilities
# 
class Seq_NN_Ensemble_By_Probabilities(nn.Module):
    """
    MultiLayer Perceptron Ensemble that combines the output 
    probabilities of n_channel MLP networks

    Constructor Parameters:
           n_channels (=14, default): number of sensors or images for each case
           n_features(=40, default): number of features for each n_channels
           n_classes(=3, by default): number of output classes 
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)


    """

    def __init__(self, n_channels=14, n_features=40, n_classes=3, 
                 hidden=[128],p_drop=0.1,relu_config=None,batch_config=None):
        super().__init__()

        self.hidden=hidden.copy()
        self.n_nets = n_channels
        self.n_features=n_features
        self.n_classes=n_classes
        self.sensors = np.arange(n_channels)

        
      #  print('running class ', self.__class__.__name__) # agregado
      #  print('Input Sensors----> ', self.sensors) # agregado
      
        # Generate n_nets classification networks (self.model_arr), one for each input sensor
        self.model_arr = nn.ModuleList([ ])
        for i in range(self.n_nets):
            self.model_arr.append(linear_block(n_features, 
                                               self.hidden.copy(), n_classes,
                                                relu_config=relu_config, 
                                                batch_config=batch_config,
                                                p_drop_loc=p_drop))
        # Agregate self.model_arr output to be the input of the 
        # network that fuses all sensors
        self.fc_out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.n_nets * n_classes, n_classes),
            nn.Sigmoid(),
            )

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
      
        out = []
        for i in range(self.n_nets):
            data = x[:,i,:]
            data = self.model_arr[i](data) # the best at 200 epochs
            out.append(data)

        out = torch.stack(out, dim=1) # [N, models, outs]
        out = out.view(out.size(0), -1) # [N, models * outs]
        out = self.fc_out(out)
        return out

# =============================================================================

class MY_ResNet(nn.Module):
    """
    Input data is [N, features=14, timestep=40]

    Architecture : Residual
        https://arxiv.org/pdf/1611.06455.pdf

    """
    def __init__(self, n_features=14, n_classes=3):
        super().__init__()

        print('running class ', self.__class__.__name__)

        n_filters = 128 # the best at 128, block of 2
        self.block_1 = My_ResNetBlock(n_features, n_filters)
        self.block_2 = My_ResNetBlock(n_filters, n_filters * 2)
        self.block_3 = My_ResNetBlock(n_filters * 2, n_filters * 2)

        self.fc_out = nn.Sequential(
            nn.Linear(n_filters * 2, n_classes),
        )

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        k_size = x.shape[2]
        x = F.avg_pool1d(x, k_size)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x












