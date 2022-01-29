#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:34:33 2021

@author: johnviljoen
"""

import torch
import torch.nn as nn
import numpy as np
from ctypes import CDLL
import ctypes

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import ToTensor

from tables import hifi_C, hifi_damping, hifi_C_lef, hifi_damping_lef, \
    hifi_rudder, hifi_ailerons, hifi_other_coeffs
    
import matplotlib.pyplot as plt
    
dtype = torch.float64

class dataset(torch.utils.data.Dataset):
    
    '''
    PyTorch dataset for learning the lookup tables of the F16, a new isntance
    of this class is required for each seperate table to be learned.
    '''
    
    def __init__(self, coeff, datapoints):
        
        self.coeff = coeff
        
        # select datapoints randomly rather than linspace
        alpha = torch.rand(datapoints, dtype=dtype)*110 - 20
        beta = torch.rand(datapoints, dtype=dtype)*60 - 30
        el = torch.rand(datapoints, dtype=dtype)*50 - 25
        
        alpha = torch.linspace(-20, 90, datapoints, dtype=dtype)
        beta = torch.linspace(-30, 30, datapoints,dtype=dtype)
        el = torch.linspace(-25, 25, datapoints,dtype=dtype)
        
        alpha = torch.reshape(alpha, [1,datapoints])
        beta = torch.reshape(beta, [1,datapoints])
        el = torch.reshape(el, [1,datapoints])
        
        
        
        # find lookup table for requested coefficient
        if self.coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
            self.targ_func = hifi_C
            self.inputs_tensor = torch.cat([alpha, beta, el])
        elif self.coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
            self.targ_func = hifi_damping
            self.inputs_tensor = alpha
        elif self.coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
            self.targ_func = hifi_C_lef
            self.inputs_tensor = torch.cat([alpha, beta])
        elif self.coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
            self.targ_func = hifi_damping_lef
            self.inputs_tensor = alpha
        elif self.coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
            self.targ_func = hifi_rudder
            self.inputs_tensor = torch.cat([alpha, beta])
        elif self.coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
            self.targ_func = hifi_ailerons
            self.inputs_tensor = torch.cat([alpha, beta])
        elif self.coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
            self.targ_func = hifi_other_coeffs
            self.inputs_tensor = torch.cat([alpha, el])

        
        
    def __len__(self):
        return self.inputs_tensor.shape[1]
    
    def __getitem__(self, idx):
        
        inputs = self.inputs_tensor[:,idx]
        
        if self.targ_func == hifi_C:
            Cx, Cz, Cm, Cy, Cn, Cl = self.targ_func(inputs[0], inputs[1], inputs[2])
        elif self.targ_func == hifi_damping:
            Cxq, Cyr, Cyp, Czq, Clr, Clp, Cmq, Cnr, Cnp = self.targ_func(inputs[0])
        elif self.targ_func == hifi_C_lef:
            delta_Cx_lef, delta_Cz_lef, delta_Cm_lef, delta_Cy_lef, delta_Cn_lef, \
                delta_Cl_lef = self.targ_func(inputs[0],inputs[1])
        elif self.targ_func == hifi_damping_lef:
            delta_Cxq_lef, delta_Cyr_lef, delta_Cyp_lef, delta_Czq_lef, \
                delta_Clr_lef, delta_Clp_lef, delta_Cmq_lef, delta_Cnr_lef, \
                    delta_Cnp_lef = self.targ_func(inputs[0])
        elif self.targ_func == hifi_rudder:
            delta_Cy_r30, delta_Cn_r30, delta_Cl_r30 = self.targ_func(inputs[0],inputs[1])
        elif self.targ_func == hifi_ailerons:
            delta_Cy_a20, delta_Cy_a20_lef, delta_Cn_a20, delta_Cn_a20_lef, \
                delta_Cl_a20, delta_Cl_a20_lef = self.targ_func(inputs[0],inputs[1])
        elif self.targ_func == hifi_other_coeffs:
            delta_Cnbeta, delta_Clbeta, delta_Cm, eta_el, delta_Cm_ds = self.targ_func(inputs[0],inputs[2])
        
        # select correct output
        target = eval(self.coeff)
        
        return inputs, target

def train_regressor_3d(dataset):
    
    train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    regressor_3d = nn.Sequential(nn.Linear(3,1024),
                                 nn.Tanh(),
                                 nn.Linear(1024,1024),
                                 nn.Tanh(),
                                 nn.Linear(1024,1024),
                                 nn.Tanh(),
                                 nn.Linear(1024,1)).double()
    
    loss_func = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(regressor_3d.parameters(),lr=0.01)
    
    losses = []

    for idx, (inputs, target) in enumerate(train_dataloader):
        
        print(f'target.shape: {target.shape}')
        print(f'inputs.shape: {inputs.shape}')
        
        regressor_3d.train()
        output = regressor_3d(inputs)#.squeeze())
        
        loss = loss_func(target.squeeze(), output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        
        print(loss)
        
    plt.plot(losses)
    
    return regressor_3d

def train_regressor_1d(dataset):
    t
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    regressor_1d = nn.Sequential(nn.Linear(1,512),
                                  # nn.Tanh(),
                                  torch.nn.ReLU(),
                                  nn.Linear(512,512),
                                  torch.nn.ReLU(),
                                  nn.Linear(512,512),
                                  torch.nn.ReLU(),
                                  nn.Linear(512,512),
                                  # nn.Tanh(),
                                  nn.Linear(512,1),
                                  ).double()
    
    # N, D_in, H, D_out = 128, 1, 1024, 1
    # regressor_1d = torch.nn.Sequential(
    #                             torch.nn.Linear(D_in, H),
    #                             torch.nn.ReLU(),
    #                             torch.nn.Linear(H, D_out),
    #                             torch.nn.Sigmoid()
    #                         ).double()
    # loss_func = torch.nn.BCELoss()
    loss_func = nn.MSELoss(reduction='mean')
    # loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(regressor_1d.parameters(),lr=0.001)
    
    losses = []
    
    for idx, (inputs, target) in enumerate(train_dataloader):
        
        print(f'target.shape: {target.shape}')
        print(f'inputs.shape: {inputs.shape}')
        
        inputs = (inputs+20) / 110 
        target = target
        
        regressor_1d.train()
        output = regressor_1d(inputs)#.squeeze())
        
        loss = loss_func(target.squeeze(), output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        
        print(loss)
        
    return regressor_1d
    # plt.plot(losses)
    
Cx_dataset = dataset('Cl', 512*100) # must be a multiple of batch size for robustness
Cnp_dataset = dataset('Cnp', 512*100) # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NN = train_regressor_1d(Cnp_dataset)

alpha= torch.linspace(-20,90,100, dtype=dtype)
Cnp_storage = np.zeros(100)
Cnp_NN_storage = np.zeros(100)
for i in range(alpha.shape[0]):
    Cxq, Cyr, Cyp, Czq, Clr, Clp, Cmq, Cnr, Cnp = hifi_damping(alpha[i])
    Cnp_storage[i] = Cnp
    Cnp_NN_storage[i] = NN.forward(alpha[i].reshape([1,1]))
    
    
    
    
plt.plot(alpha,Cnp_storage)
plt.plot(alpha,Cnp_NN_storage)