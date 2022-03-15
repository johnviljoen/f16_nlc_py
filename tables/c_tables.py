"""
This script wraps the C shared library .so file

NOTE: This script MUST be run from the directory above this one, as the C file
expects to reach aerodata through the following path:

    tables/aerodata

If this script is run from this directory a segmentation fault WILL occur.

NOTE: This script could be further accelerated by retaining prior pointers
to prior xdots from prior lookups, but this has not yet been implemented.
"""

import torch
import ctypes
import os
import numpy as np
import sys

aerodata_path = "/home/jovi/Documents/Code/f16_pt/tables/aerodata"

dtype = torch.double
tables = ctypes.CDLL(aerodata_path + "/hifi_F16_AeroData.so")

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

c_lookup = C_lookup()