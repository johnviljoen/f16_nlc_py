#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:02:06 2020

@author: johnviljoen
"""

''' This file contains all parameters (paras) required to run the simulation,
the aircraft, environmental, simulation, initial conditions, and other parameters'''

#import numpy as np
import numpy as np
import torch
from numpy import pi
from scipy.constants import g
import os
from ctypes import CDLL
from dataclasses import dataclass

# In[simulation parameters]  

dt, time_start, time_end = 0.001, 0., 10.

# fi_flag = 1 -> high fidelity model (full Nguyen)
# fi_flag = 1 -> low fidelity model (Stevens Lewis reduced)
fi_flag = 1

# stability_flag only functional for high fidelity model currently!
# stability_flag = 1 -> unstable xcg 35% model
# stability_flag = 0 -> stable xcg 25% model
stab_flag = 0

# In[initial_conditions]  

''' states in m/s, rad, rad/s '''
npos        = 0.                # m
epos        = 0.                # m
h           = 3048.             # m
phi         = 0.                # rad
theta       = 0.                # rad
psi         = 0.                # rad

vt          = 213.36            # m/s
alpha       = 1.0721 * pi/180   # rad
beta        = 0.                # rad
p           = 0.                # rad/s
q           = 0.                # rad/s
r           = 0.                # rad/s

''' control states in lbs, deg '''
T           = 2886.6468         # lbs
dh          = -2.0385           # deg
da          = -0.087577         # deg
dr          = -0.03877          # deg
lef         = 0.3986            # deg

# In[limits]

npos_min        = -np.inf       # (m)
epos_min        = -np.inf       # (m)
h_min           = 0             # (m)
phi_min         = -np.inf       # (deg)
theta_min       = -np.inf       # (deg)
psi_min         = -np.inf       # (deg)
V_min           = 0             # (m/s)
alpha_min       = -20.          # (deg)
beta_min        = -30.          # (deg)
p_min           = -300          # (deg/s)
q_min           = -100          # (deg/s)
r_min           = -50           # (deg/s)

T_min           = 1000          # (lbs)
dh_min          = -25           # (deg)
da_min          = -21.5         # (deg)
dr_min          = -30.          # (deg)
lef_min         = 0.            # (deg)

npos_max        = np.inf        # (m)
epos_max        = np.inf        # (m)
h_max           = 100000        # (m)
phi_max         = np.inf        # (deg)
theta_max       = np.inf        # (deg)
psi_max         = np.inf        # (deg)
V_max           = 900           # (m/s)
alpha_max       = 90            # (deg)
beta_max        = 30            # (deg)
p_max           = 300           # (deg/s)
q_max           = 100           # (deg/s)
r_max           = 50            # (deg/s)

T_max           = 19000         # (lbs)
dh_max          = 25            # (deg)
da_max          = 21.5          # (deg)
dr_max          = 30            # (deg)
lef_max         = 25            # (deg)

# In[wrap for input]  

# initial_state_vector = np.array([npos, epos, h, phi, theta, psi, vt, alpha, beta, p, q, r, T, dh, da, dr, lef, fi_flag])
# model_predictive_control_parameters = [hzn, pred_dt]

m2f = 3.28084 # metres to feet conversion
f2m = 1/m2f # feet to metres conversion

x0 = torch.tensor([npos*m2f, epos*m2f, h*m2f, phi, theta, psi, vt*m2f, alpha, beta, p, q, r, T, dh, da, dr, lef, -alpha*180/pi])#[np.newaxis].T
u0 = torch.clone(x0[12:16])

#if stab_flag == 1:
#    so_file = os.getcwd() + "/dynamics/C/nlplant_xcg35.so"
#elif stab_flag == 0:
#    so_file = os.getcwd() + "/dynamics/C/nlplant_xcg25.so"
#    
#so_file = os.getcwd() + "/dynamics/C_old/nlplant.so"
## so_file = os.getcwd() + "/C/nlplant.so"
#nlplant = CDLL(so_file)

states = ['npos','epos','h','phi','theta','psi','V','alpha','beta','p','q','r','T','dh','da','dr','lf2','lf1']
inputs = ['T','dh','da','dr']

x_units = ['ft','ft','ft','rad','rad','rad','ft/s','rad','rad','rad/s','rad/s','rad/s','lb','deg','deg','deg','deg','deg']
u_units = ['lb','deg','deg','deg']

x_ub = [npos_max, epos_max, h_max, phi_max, theta_max, psi_max, V_max, alpha_max, beta_max, p_max, q_max, r_max, T_max, dh_max, da_max, dr_max, lef_max, np.infty]
x_lb = [npos_min, epos_min, h_min, phi_min, theta_min, psi_min, V_min, alpha_min, beta_min, p_min, q_min, r_min, T_min, dh_min, da_min, dr_min, lef_min, -np.infty]

u_ub = [T_max, dh_max, da_max, dr_max]
u_lb = [T_min, dh_min, da_min, dr_min]

udot_ub = [10000, 60, 80, 120]
udot_lb = [-10000, -60, -80, -120]

# In[mpc control choices] -> must be in order!

# observed_states = ['V','alpha','beta','p','q','r']
observed_states = ['h','phi','theta','alpha','beta','p','q','r','lf2','lf1']
mpc_states = ['phi','theta','alpha','beta','p','q','r','lf1','lf2']
mpc_inputs = ['dh','da','dr']
mpc_controlled_states = ['p','q','r']

# In[dataclass wrap]

@dataclass
class stateVector:
    states: list
    values: torch.tensor
    units: list
    upper_bound: list
    lower_bound: list
    initial_condition: torch.tensor
    observed_states: list
    mpc_states: list
    mpc_inputs: list
    mpc_controlled_states: list
    _obs_x_idx: list = None
    _mpc_x_idx: list = None
    _mpc_x_lb: list = None
    _mpc_x_ub: list = None
    
    def __post_init__(self):
        
        self._obs_x_idx = [self.states.index(self.observed_states[i]) for i in range(len(self.observed_states)) if self.observed_states[i] in self.states]
        self._mpc_x_idx = [self.states.index(self.mpc_states[i]) for i in range(len(self.mpc_states)) if self.mpc_states[i] in self.states]
        
        self._mpc_u_states_idx = [self.states.index(self.mpc_inputs[i]) for i in range(len(self.mpc_inputs)) if self.mpc_inputs[i] in self.states]        
        self._mpc_u_in_mpc_x_idx = [self.mpc_states.index(self.mpc_controlled_states[i]) for i in range(len(self.mpc_controlled_states)) if self.mpc_controlled_states[i] in self.mpc_states]
        self._mpc_u_in_x_idx = [self.states.index(self.mpc_inputs[i]) for i in range(len(self.mpc_inputs)) if self.mpc_inputs[i] in self.states]
        
        
        self._mpc_x_lb = [self.lower_bound[i] for i in self._mpc_x_idx]
        self._mpc_x_ub = [self.upper_bound[i] for i in self._mpc_x_idx]
        
        self._mpc_obs_x_idx = [i for i in range(len(self.mpc_states)) if self.mpc_states[i] in self.observed_states]
                
        self._np_x_lb = torch.tensor(self.lower_bound)
        self._np_x_ub = torch.tensor(self.upper_bound)
        
        self._vec_mpc_x_lb = torch.tensor(self._mpc_x_lb)[:,None]
        self._vec_mpc_x_ub = torch.tensor(self._mpc_x_ub)[:,None]

    def __call__(self):
        return self.values
        
    def _get_mpc_x(self):
        return torch.tensor([self.values[i] for i in self._mpc_x_idx])
    
    def _get_mpc_act_states(self):
        return torch.tensor([self.values[i] for i in self._mpc_u_states_idx])
    
        
@dataclass
class inputVector:
    inputs: list
    values: torch.tensor
    units: list
    upper_cmd_bound: list
    lower_cmd_bound: list
    upper_rate_bound: list
    lower_rate_bound: list
    initial_condition: torch.tensor
    mpc_inputs: list
    
    def __post_init__(self):
        self._mpc_u_idx = [self.inputs.index(self.mpc_inputs[i]) for i in range(len(mpc_inputs)) if self.mpc_inputs[i] in self.inputs]
        self._mpc_u_lb = [self.lower_cmd_bound[i] for i in self._mpc_u_idx]
        self._mpc_u_ub = [self.upper_cmd_bound[i] for i in self._mpc_u_idx]
        self._mpc_udot_lb = [self.lower_rate_bound[i] for i in self._mpc_u_idx]
        self._mpc_udot_ub = [self.upper_rate_bound[i] for i in self._mpc_u_idx]
        self._vec_mpc_u_lb = torch.tensor(self._mpc_u_lb).unsqueeze(0).T
        self._vec_mpc_u_ub = torch.tensor(self._mpc_u_ub).unsqueeze(0).T
        self._vec_mpc_udot_lb = torch.tensor(self._mpc_udot_lb).unsqueeze(0).T
        self._vec_mpc_udot_ub = torch.tensor(self._mpc_udot_ub).unsqueeze(0).T
    
    def __call__(self):
        return self.values
        
    def _get_mpc_u(self):
        return torch.tensor([self.values[i] for i in self._mpc_u_idx])
    
@dataclass#(frozen=True)
class simulationParameters:
    dt: float
    time_start: float
    time_end: float
    stab_flag: int
    fi_flag: int
    
@dataclass
class SS:
    A: torch.tensor
    B: torch.tensor
    C: torch.tensor
    D: torch.tensor
    x_lin: torch.tensor
    u_lin: torch.tensor
    dt: float
    

state_vector = stateVector(
    states,
    torch.clone(x0),
    x_units,
    x_ub,
    x_lb,
    torch.clone(x0),
    observed_states,
    mpc_states,
    mpc_inputs,
    mpc_controlled_states)    

input_vector = inputVector(
    inputs,
    torch.clone(u0),
    u_units,
    u_ub,
    u_lb,
    udot_ub,
    udot_lb,
    torch.clone(u0),
    mpc_inputs)       
       
simulation_parameters = simulationParameters(
    dt,
    time_start,
    time_end,
    stab_flag,
    fi_flag)

# In[additional info provided for brevity]

# weight                  = 91188         # Newtons

# Ixx                     = 12875         # Kg m^2
# Iyy                     = 75674         # Kg m^2
# Izz                     = 85552         # Kg m^2
# Ixz                     = 1331          # Kg m^2
# # the other Izy, Iyz = 0

# b                       = 9.144         # m wingspan
# S                       = 27.87         # m^2 wing area
# cbar                    = 3.45          # m wing mean aerodynamic chord

# He                      = 216.9         # engine angular momentum constant

# x_cg_ref                = 0.35 * cbar   # assuming mac = cbar
# x_cg                    = 0.8*x_cg_ref  # FOR NOW THIS IS WRONG

# # unecessary:
# length = 14.8 #m
# height = 4.8 #m
