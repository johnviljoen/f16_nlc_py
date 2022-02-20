#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:09:43 2021

@author: johnviljoen
"""
import os
import torch
import numpy as np
import ctypes
from ctypes import CDLL
from scipy.interpolate import LinearNDInterpolator

import scipy
tables = CDLL('dynamics/C/hifi_F16_AeroData.so')
tables =  CDLL('../f16_pt_29-01-2022/C/hifi_F16_AeroData.so')
dtype = torch.float64

class C_lookup():

    def __init__(self):
        pass

    def hifi_C(self, inp):
        

        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        beta_compat = ctypes.c_double(float(inp[1].numpy()))
        el_compat = ctypes.c_double(float(inp[2].numpy()))

        
        tables.hifi_C(alpha_compat, beta_compat, el_compat, retVal_pointer)
        
        return torch.tensor(retVal) # Cx, Cz, Cm, Cy, Cn, Cl

    def hifi_damping(self, inp):
        
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        
        tables.hifi_damping(alpha_compat, retVal_pointer)

        return torch.tensor(retVal, dtype=dtype)

    def hifi_C_lef(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        beta_compat = ctypes.c_double(float(inp[1].numpy()))
        
        tables.hifi_C_lef(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)
        obsv
        ''' This table only accepts alpha up to 45 '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
       
        
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        
        tables.hifi_damping_lef(alpha_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_rudder(self, inp):
        
        retVal = np.zeros(3)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        beta_compat = ctypes.c_double(float(inp[1].numpy()))
        
        tables.hifi_rudder(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_ailerons(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        beta_compat = ctypes.c_double(float(inp[1].numpy()))
        
        tables.hifi_ailerons(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_other_coeffs(self, inp):
        
        '''expects an input of alpha, el'''
        
        retVal = np.zeros(5)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].numpy()))
        el_compat = ctypes.c_double(float(inp[1].numpy()))
        
        tables.hifi_other_coeffs(alpha_compat, el_compat, retVal_pointer)
        
        retVal[4] = 0 # ignore deep-stall regime, delta_Cm_ds = 0
        
        return torch.tensor(retVal, dtype=dtype)

class Py_lookup():
    def __init__(self):
        # indices lookup
        self.axes = {}
        self.axes['ALPHA1'] = torch.tensor(self.read_file("tables/aerodata/ALPHA1.dat"))
        self.axes['ALPHA2'] = torch.tensor(self.read_file("tables/aerodata/ALPHA2.dat"))
        self.axes['BETA1'] = torch.tensor(self.read_file("tables/aerodata/BETA1.dat"))
        self.axes['DH1'] = torch.tensor(self.read_file("tables/aerodata/DH1.dat"))
        self.axes['DH2'] = torch.tensor(self.read_file("tables/aerodata/DH2.dat"))
        
        # tables store the actual data, points are the alpha, beta, dh axes 
        self.tables = {}
        self.points = {}
        self.ndinfo = {}
        for file in os.listdir("tables/aerodata"):
            alpha_len = None
            beta_len = None
            dh_len = None
            alpha_fi = None
            beta_fi = None
            dh_fi = None
            if "_ALPHA1" in file:
                alpha_len = len(self.axes['ALPHA1'])
                alpha_fi = 'ALPHA1'
            if "_ALPHA2" in file:
                alpha_len = len(self.axes['ALPHA2'])
                alpha_fi = 'ALPHA2'
            if "_BETA1" in file:
                beta_len = len(self.axes['BETA1'])
                beta_fi = 'BETA1'
            if "_DH1" in file:
                dh_len = len(self.axes['DH1'])
                dh_fi = 'DH1'
            if "_DH2" in file:
                dh_len = len(self.axes['DH2'])
                dh_fi = 'DH2'

            temp = [alpha_len, beta_len, dh_len]
            dims = [i for i in temp if i is not None]

            # 1D tables
            if len(dims) == 1:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}"))
            
            # 2D tables
            elif len(dims) == 2:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}")).reshape([dims[0],dims[1]])
            
            # 3D tables
            elif len(dims) == 3:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}")).reshape([dims[0],dims[1],dims[2]])

            self.ndinfo[file] = {
                'alpha_fi': alpha_fi,
                'beta_fi': beta_fi,
                'dh_fi': dh_fi
            }
        
        key = 'CN0120_ALPHA1_BETA1_DH2_501.dat'
        self.tables[key]

        xi = torch.tensor([1.,2.,0.]).numpy()
        self.interpn(key, xi)
        import pdb
        pdb.set_trace()



    def read_file(self, path):
        
        # get the indices of the various tables first
        with open(path) as f:
            lines = f.readlines()
        temp = lines[0][:-1].split()
        line = [float(i) for i in temp]
        return line

    def interpn(self, key, xi):
        
        points = (
            self.axes[self.ndinfo[key]['alpha_fi']].numpy(), 
            self.axes[self.ndinfo[key]['beta_fi']].numpy(),
            self.axes[self.ndinfo[key]['dh_fi']].numpy())
        values = self.tables[key].numpy()
        interp = torch.tensor(scipy.interpolate.interpn(points,values,xi))
        return interp 

    def hifi_C(self, inp):
        pass
        
Py_table = Py_lookup()

table_C = C_lookup()
class table_wrap():

    def __init__(self, coeff):

        self.coeff = coeff
        C = C_lookup()
        # find lookup table for requested coefficient
        if self.coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
            self.table = C.hifi_C
            self.table_outputs = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
        elif self.coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
            self.table = C.hifi_damping
            self.table_outputs = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
        elif self.coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
            self.table = C.hifi_C_lef
            self.table_outputs = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
        elif self.coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
            self.table = C.hifi_damping_lef
            self.table_outputs = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
        elif self.coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
            self.table = C.hifi_rudder
            self.table_outputs = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
        elif self.coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
            self.table = C.hifi_ailerons
            self.table_outputs = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
        elif self.coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
            self.table = C.hifi_other_coeffs
            self.table_outputs = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']

    def call(self, inp):
        # select the correct table and extract correct output
        table_output_idx = self.table_outputs.index(self.coeff)

        return self.table(inp)[table_output_idx]


#class table_wrap():
#    
#    def __init__(self, coeff):
#        
#        self.coeff = coeff
#        
#        # find lookup table for requested coefficient
#        if self.coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
#            self.table = hifi_C
#            self.table_outputs = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
#        elif self.coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
#            self.table = hifi_damping
#            self.table_outputs = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
#        elif self.coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
#            self.table = hifi_C_lef
#            self.table_outputs = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
#        elif self.coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
#            self.table = hifi_damping_lef
#            self.table_outputs = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
#        elif self.coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
#            self.table = hifi_rudder
#            self.table_outputs = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
#        elif self.coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
#            self.table = hifi_ailerons
#            self.table_outputs = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
#        elif self.coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
#            self.table = hifi_other_coeffs
#            self.table_outputs = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']
#            
#    def call(self, inp):
#        # select the correct table and extract correct output
#        table_output_idx = self.table_outputs.index(self.coeff)
#
#        return self.table(inp)[table_output_idx]
