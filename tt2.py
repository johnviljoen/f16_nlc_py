#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:17:33 2021

@author: johnviljoen
"""

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from tables.c_tables import C_lookup, table_wrap
from tables.NN import approximator

# In[settings]

def train_table(coeff, LR, device, dtype, coeff_lim,
                train_datapoints=80000,
                val_datapoints=20000,
                MAX_EPOCH=100,
                BATCH_SIZE=512,
                ):
    
    # In[]
    
    model = approximator(coeff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss(reduction="mean")
    
    # In[]
    
    inp_size = len(model.input_names)
    
    # Create table instance
    table = table_wrap(coeff)
    
    # First input randomisation -> can change to gaussian to focus training
    alpha_train = torch.rand([train_datapoints, 1]) * 110 - 20
    beta_train = torch.rand([train_datapoints, 1]) * 60 - 30
    el_train = torch.rand([train_datapoints, 1]) * 50 - 25
    X_train = torch.cat([alpha_train, beta_train, el_train], 1)
    
    alpha_val = torch.rand([val_datapoints, 1]) * 110 - 20
    beta_val = torch.rand([val_datapoints, 1]) * 60 - 30
    el_val = torch.rand([val_datapoints, 1]) * 50 - 25
    X_val = torch.cat([alpha_val, beta_val, el_val], 1)
    
    if model.input_names == ['alpha']:
        X_train = alpha_train
        X_val = alpha_val
    elif model.input_names == ['alpha', 'beta']:
        X_train = torch.cat([alpha_train, beta_train], 1)
        X_val = torch.cat([alpha_val, beta_val], 1)
    elif model.input_names == ['alpha', 'el']:
        X_train = torch.cat([alpha_train, el_train], 1)
        X_val = torch.cat([alpha_val, el_val], 1)
    elif model.input_names == ['alpha', 'beta', 'el']:
        X_train = torch.cat([alpha_train, beta_train, el_train], 1)
        X_val = torch.cat([alpha_val, beta_val, el_val], 1)
    
    y_train = torch.zeros([train_datapoints, 1])
    for i in range(train_datapoints):
        y_train[i] = table.call(X_train[i])
    
    y_val = torch.zeros([val_datapoints, 1])
    for i in range(val_datapoints):
        y_val[i] = table.call(X_val[i])
        

        
    # normalise the X_train, X_val
    X_train = (X_train + 20) / 110
    X_val = (X_val + 20) / 110
    
    def normalise(tensor, maximum, minimum):
        return (tensor - minimum) / (maximum - minimum)
    
    def denormalise(tensor, maximum, minimum):
        return tensor * (maximum - minimum) + minimum
    
    y_train = normalise(y_train, coeff_lim[coeff][0], coeff_lim[coeff][1])
    y_val = normalise(y_val, coeff_lim[coeff][0], coeff_lim[coeff][1])
    
    # Use TensorDataset to hold training and validation io in memory
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    # Use DataLoader for further randomisation
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  pin_memory=True, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                pin_memory=True, shuffle=True)
    
    # scheduler does work, but sometimes catches it at a high cost point and it stays there
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True, min_lr=1e-5)
    
    # In[]
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in tqdm(range(MAX_EPOCH)):
        model.train()
        
        temp_loss_list = []
        for X_train, y_train in train_dataloader:
            import pdb
            pdb.set_trace()
            X_train = X_train.type(dtype).to(device)
            y_train = y_train.type(dtype).to(device)

            
            optimizer.zero_grad()
            
            output = model(X_train)
            loss = loss_func(input=output, target=y_train)
            loss.backward()
            
            optimizer.step()
            
            temp_loss_list.append(loss.detach().cpu().numpy())
            
        temp_loss_list = []
        
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(dtype).to(device)
            y_train = y_train.type(dtype).to(device)
            
            output = model(X_train)
            loss = loss_func(input=output, target=y_train)
            
            temp_loss_list.append(loss.detach().cpu().numpy())
            
        train_loss_list.append(np.average(temp_loss_list))
        
        model.eval()
        
        temp_loss_list = []
        for X_val, y_val in val_dataloader:
            X_val = X_val.type(dtype).to(device)
            y_val = y_val.type(dtype).to(device)
    
            score = model(X_val)
            loss = loss_func(input=score, target=y_val)
    
            temp_loss_list.append(loss.detach().cpu().numpy())
        
        val_loss_list.append(np.average(temp_loss_list))
        
        scheduler.step(torch.tensor(np.average(temp_loss_list)))
    
        print("\n \ttrain loss: %.5f" % train_loss_list[-1])
        print("\tval loss: %.5f" % val_loss_list[-1])
        
        # save best model
        if epoch == val_loss_list.index(min(val_loss_list)):
            
            print('saving best model')
            torch.save({'model':model}, f'tables/NNs/{coeff}.pt')
        
    return model, train_loss_list, val_loss_list
    
    
# In[]

if __name__ == '__main__':
    
    list1 = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
    list2 = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
    list3 = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
    list4 = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
    list5 = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
    list6 = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
    list7 = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']

    full_list = list1 + list2 + list3 + list4 + list5 + list6 + list7
    full_list = ['Clr']

    list_1d = list2 + list4
    list_2d = list3 + list5 + list6 + list7
    list_3d = list1
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    coeff_lim = torch.load('tables/NNs/lookup_table_limits.pt')
    # coeff = 'Cxq'
    LR = 1e-3
    train_datapoints = 80000
    val_datapoints = 20000
    MAX_EPOCH = 1000
    BATCH_SIZE = 512
    
    # failure:
    test_list = ['Clr']
    # select which tables to train and update the NNs folder with
    for i, coeff in enumerate(full_list):
    
        model, train_loss_list, val_loss_list = train_table(coeff, LR, device, dtype, coeff_lim,
                        train_datapoints=train_datapoints,
                        val_datapoints=val_datapoints,
                        MAX_EPOCH=MAX_EPOCH,
                        BATCH_SIZE=BATCH_SIZE,
                        )
        
        # torch.save({'model':model, 'training loss':train_loss_list, 'validation loss':val_loss_list}, f'NNs/{coeff}.pt')
        
        fig1 = plt.figure(figsize=(6,4))
        plt.semilogy(train_loss_list)
        plt.semilogy(val_loss_list)
        plt.legend(['train_loss', 'validation_loss'])
        plt.savefig(f'NNs/{coeff}_loss.png')
        plt.close(fig1)
        
        with open('NNs/loss_manifest.txt','a') as f:
            f.write(f'{coeff}: \n')
            f.write(f'final training loss: {train_loss_list[-1]} \n')
            f.write(f'final validation loss: {val_loss_list[-1]} \n \n')
        
