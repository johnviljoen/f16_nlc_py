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
dtype = torch.float64


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
            
            self.ndinfo[file] = {
                'alpha_fi': alpha_fi,
                'beta_fi': beta_fi,
                'dh_fi': dh_fi
            }

            # 1D tables
            if len(dims) == 1:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}"))
                if file == "ETA_DH1_brett.dat":
                    self.points[file] = (self.axes[self.ndinfo[file]['dh_fi']]) 
                else:
                    self.points[file] = (self.axes[self.ndinfo[file]['alpha_fi']]) 
                    
         
            # 2D tables
            elif len(dims) == 2:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}")).reshape([dims[0],dims[1]])
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']])
                 
            # 3D tables
            elif len(dims) == 3:
                self.tables[file] = torch.tensor(self.read_file(f"tables/aerodata/{file}")).reshape([dims[0],dims[1],dims[2]])
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']],
                    self.axes[self.ndinfo[file]['dh_fi']]) 


        
        key = 'CN0120_ALPHA1_BETA1_DH2_501.dat'
        self.tables[key]

        xi = torch.tensor([1.,2.,0.]).numpy()
       # #self.interpn(key, xi)
       # Cx = self.Cx(0.,1.0,0.0) 
       # Cz = self.Cz(0.,1.0,0.0) 
       # Cm = self.Cm(0.,1.0,0.0) 
       # Cy = self.Cy(0.,1.0) 
       # Cn = self.Cn(0.,1.0,0.0) 
       # Cl = self.Cl(0.,1.0,0.0) 
       # 
       # Cx_lef = self.Cx_lef(0.,1.0) 
       # Cz_lef = self.Cz_lef(0.,1.0) 
       # Cm_lef = self.Cm_lef(0.,1.0) 
       # Cy_lef = self.Cy_lef(0.,1.0) 
       # Cn_lef = self.Cn_lef(0.,1.0) 
       # Cl_lef = self.Cl_lef(0.,1.0)

        #CXq = self.CXq(1.0)
        key = "CX1120_ALPHA1_204.dat"
        xi = torch.tensor(10.) # 10 degrees alpha
       
        self.interp1d(key, xi)




    def read_file(self, path):
        """
        Utility for reading in the .dat files that comprise all of the aerodata
        """
        
        # get the indices of the various tables first
        with open(path) as f:
            lines = f.readlines()
        temp = lines[0][:-1].split()
        line = [float(i) for i in temp]
        return line

    def interpn(self, key, xi):
        """
        Interpolates the queried point in the value table defined by tables and points.
        It does require scipy and torch -> numpy -> torch conversion so could be optimised
        """
        import pdb
        pdb.set_trace()
        return torch.tensor(scipy.interpolate.interpn(
            self.points[key],
            self.tables[key].numpy(),
            xi))

    def index(self, tensor, value):
        """
        Utility for finding index of value in tensor
        """
        return ((tensor == value).nonzero(as_tuple=True)[0])

    def interp1d(self, key, xi):
        """
        Uses a custom interpolation function for the lookup tables as scipys implementation
        requires conversion to numpy and back. I wish to keep everything in pytorch for
        speed.
        """
        # step 1: find the two known datapoints the query point is between
        try:        
            pass 

        except: # if the value to be queried lies exactly on the known datapoint we will except
            return torch.index_select(self.tables[key], 0, self.index(self.points[key], xi))

    # hifi_C: Cx, Cz, Cm, Cy, Cn, Cl
    def Cx(self, alpha, beta, dh):
        return self.interpn("CX0120_ALPHA1_BETA1_DH1_201.dat", np.array([alpha, beta, dh]))
    def Cz(self, alpha, beta, dh):
        return self.interpn("CZ0120_ALPHA1_BETA1_DH1_301.dat", np.array([alpha, beta, dh]))
    def Cm(self, alpha, beta, dh):
        return self.interpn("CM0120_ALPHA1_BETA1_DH1_101.dat", np.array([alpha, beta, dh]))
    def Cy(self, alpha, beta):
        return self.interpn("CY0320_ALPHA1_BETA1_401.dat", np.array([alpha, beta]))
    def Cn(self, alpha, beta, dh):
        return self.interpn("CN0120_ALPHA1_BETA1_DH2_501.dat", np.array([alpha, beta, dh]))
    def Cl(self, alpha, beta, dh):
        return self.interpn("CL0120_ALPHA1_BETA1_DH2_601.dat", np.array([alpha, beta, dh]))
       
    # next
    def Cx_lef(self, alpha, beta):
        return self.interpn("CX0820_ALPHA2_BETA1_202.dat", np.array([alpha, beta]))
    def Cz_lef(self, alpha, beta):
        return self.interpn("CZ0820_ALPHA2_BETA1_302.dat", np.array([alpha, beta]))
    def Cm_lef(self, alpha, beta):
        return self.interpn("CM0820_ALPHA2_BETA1_102.dat", np.array([alpha, beta]))
    def Cy_lef(self, alpha, beta):
        return self.interpn("CY0820_ALPHA2_BETA1_402.dat", np.array([alpha, beta]))
    def Cn_lef(self, alpha, beta):
        return self.interpn("CN0820_ALPHA2_BETA1_502.dat", np.array([alpha, beta]))
    def Cl_lef(self, alpha, beta):
        return self.interpn("CL0820_ALPHA2_BETA1_602.dat", np.array([alpha, beta]))

    # next
    def CXq(self, alpha):
        return self.interpn("CX1120_ALPHA1_204.dat", np.array([alpha]))
        
        
        

py_lookup = Py_lookup()


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
