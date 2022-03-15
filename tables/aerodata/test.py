# script to test the functionality of the hifi_F16_AeroData binary
# result: It works! Dont touch it again!

import torch
import ctypes
import os
import numpy as np

dtype = torch.double
inp = torch.tensor([1.0,0.1], dtype=dtype)

inp1 = inp[0].numpy()
inp2 = inp[1].numpy()
tables = ctypes.CDLL("/home/jovi/Documents/Code/f16_pt/tables/aerodata/hifi_F16_AeroData.so")

out = np.zeros(6)
out_ptr = ctypes.c_void_p(out.ctypes.data)
#C_so.hifi_C_lef(ct.c_double(inp1), ct.c_double(inp2), out_ptr)

#def hifi_C_lef(alpha, beta):
#    alpha_compat = ct.c_double(alpha.numpy())
#    beta_compat = ct.c_double(beta.numpy())
#    out = np.zeros(6)
#    out_ptr = ct.c_void_p(out.ctypes.data)
#    C_so.hifi_C_lef(alpha_compat, beta_compat, out_ptr)
#    return out

class C_lookup():

    def __init__(self):

        inp3 = torch.tensor([0.0,0.0,0.0])
        inp2 = torch.tensor([0.,0.])
        #inp3 = np.array([0.,0.,0.])
        #inp2 = np.array([1.,5.])
        self.hifi_C_lef(inp2)
        self.hifi_C(inp3)

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
print(out)
import pdb
pdb.set_trace()