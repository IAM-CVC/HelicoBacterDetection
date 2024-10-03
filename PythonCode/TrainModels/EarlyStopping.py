#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:41:33 2022

@author: Guillermo Torres
"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warm_up=25, patience=30, delta=0):
        assert 0 <= warm_up
        assert 0 < patience
        
        self.warm_up = warm_up
        self.patience = patience
        self.counter = 0
        
        self.best_score = np.Inf   # lowest loss
        self.best_model = None
        self.epoch = 0
        
        self.early_stop = False
        self.delta = delta
        
    def __call__(self, current_epoch, current_loss, model):
        
        score = float(format(current_loss,".3f")) # set to 4 digit after the comma.
     #   print(current_epoch,score,self.best_score)
        if self.warm_up < current_epoch:
            if score < (self.best_score + self.delta):
                self.best_score = score
                self.counter = 0
                self.best_model = model   # backup model 
                self.epoch = current_epoch
            else:
                self.counter += 1            
                if self.patience <= self.counter:
                    self.early_stop = True
                    
        return self.early_stop
    
    def __del__(self):
        del self.best_model


        