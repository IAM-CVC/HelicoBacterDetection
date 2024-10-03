import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np
from random import shuffle
import random


    
# =============================================================================
# Standard dataset (Single Objective)
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.y is not None:
            return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx]))
        else:
            return torch.from_numpy(self.X[idx])

             
       
        
        
# =============================================================================
# Transformations
# =============================================================================

class IdentitySeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        return x  # [n_feats, time_steps]

class ReverseSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        x = np.ascontiguousarray(x[:, ::-1])
        return x # [n_feats, time_steps]

class ToTensorSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):

        return torch.from_numpy(x) # [n_feats, time_steps]

"""
TFG:Valorar si val la pena posar això en un .py a part
    
"""
# =============================================================================
# Data Sets Creation
# =============================================================================

def class_sample_count(labels):
    u,indices=np.unique(np.sort(labels.flatten()),return_index=True)
    nword=np.diff(np.array(list(indices)+[len(labels.flatten())]))
    return nword

def classes_weight(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(class_sample_count(y_train))
    #sample_counts = np.array(class_sample_count(y_train))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight


def classes_weight_binary(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(np.sum(y_train,axis=0))
    # avoid division by zero, and therefore, inf values in the result
 #   sample_counts[sample_counts==0] = 1
    classes_weight=1./sample_counts
    classes_weight[classes_weight==np.inf]=0
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight

def create_Paired_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=True):

#
#    if y_test.shape[0] < batch_size:
#        batch_size = y_test.shape[0]

    test_dataset = Paired_Dataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
    return test_dataloader


def create_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=True):

#
#    if y_test.shape[0] < batch_size:
#        batch_size = y_test.shape[0]

    test_dataset = Standard_Dataset(x_test, y_test, transf)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
    return test_dataloader


def create_dataloader_balanced(x_train, y_train, transf=False, batch_size=128, shuffle=False):
#    if y_train.ndim > 1:
#        print('Data was no properly encoded. Error in train_dataloader!')
#        return

    sample_counts = class_sample_count(list(y_train))
    classes_weight = 1. / torch.tensor(sample_counts, dtype=torch.float)
    samples_weight = torch.tensor([classes_weight[w] for w in y_train])

    # samples_weight=classes_weight(y_train)
    # traind dataloader
    train_dataset = Standard_Dataset(x_train, y_train, transf)
    

    # pytorch function for sampling batch based on weights or probabilities for each
    # element. To obtain a relative balaced batch, it uses replacement by default
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=shuffle, sampler=sampler)
    return train_dataloader

