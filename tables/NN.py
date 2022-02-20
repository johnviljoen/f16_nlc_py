#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:50:24 2021

@author: johnviljoen
"""

import torch.nn as nn
import torch

from tables.tables import C_lookup

C = C_lookup()



class approximator(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
        
        # minimal size 1d approximator that works well
        approximator_1d = nn.Sequential(nn.Linear(1, 6),
                                             nn.Tanh(),
                                             nn.Linear(6,6),
                                             nn.Tanh(),
                                             nn.Linear(6, 1))
        
        # another layer helps 2d functions converge much faster
        approximator_2d = nn.Sequential(nn.Linear(2, 6),
                                             nn.Tanh(),
                                             nn.Linear(6,6),
                                             nn.Tanh(),
                                             nn.Linear(6,6),
                                             nn.Tanh(),
                                             nn.Linear(6,1))
        
        # found another wider layer helps with 3d functions
        approximator_3d = nn.Sequential(nn.Linear(3,6),
                                             nn.Tanh(),
                                             nn.Linear(6,12),
                                             nn.Tanh(),
                                             nn.Linear(12, 6),
                                             nn.Tanh(),
                                             nn.Linear(6,1))
        
        # find lookup table for requested coefficient
        if self.coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
            self.table = C.hifi_C
            self.table_outputs = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
            self.model = approximator_3d
            self.input_names = ['alpha', 'beta', 'el']
            
        elif self.coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
            self.table = C.hifi_damping
            self.table_outputs = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
            self.model = approximator_1d
            self.input_names = ['alpha']
            
        elif self.coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
            self.table = C.hifi_C_lef
            self.table_outputs = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
            self.model = approximator_2d
            self.input_names = ['alpha', 'beta']
            
        elif self.coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
            self.table = C.hifi_damping_lef
            self.table_outputs = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
            self.model = approximator_1d
            self.input_names = ['alpha']
            
        elif self.coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
            self.table = C.hifi_rudder
            self.table_outputs = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
            self.model = approximator_2d
            self.input_names = ['alpha', 'beta']
            
        elif self.coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
            self.table = C.hifi_ailerons
            self.table_outputs = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
            self.model = approximator_2d
            self.input_names = ['alpha', 'beta']
            
        elif self.coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
            self.table = C.hifi_other_coeffs
            self.table_outputs = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']
            self.model = approximator_2d
            self.input_names = ['alpha', 'el']
       
    def __call__(self, inp):
        return self.forward(inp)

    def forward(self, inp):
        '''inp must be of form tensor([alpha, beta, el]) or smaller, with all values normalised'''
        # inp = (inp + 20) /110
        return self.model(inp)
        
            
        



