import torch
import torch.nn as nn
import numpy as np
from scipy.signal import cont2discrete

from dynamics.parameters import state_vector, input_vector, simulation_parameters
from dynamics.nlplant import Nlplant
from control.trim import Trim
from control.lqr import dlqr
from utils.linmod import Linmod
from utils.get_obs import Get_observation

class F16(nn.Module):
    
    def __init__(self, device, dtype):
        super().__init__()

        self.device = device
        self.dtype = dtype
        print(f'device chosen: {device}')
        print(f'dtype chosen: {dtype}')

        self.x = state_vector                   # mutable state dataclass
        self.u = input_vector                   # mutable input dataclass
        self.paras = simulation_parameters      # simulation parameters dataclass

        self.x.values = self.x.values.type(dtype).to(device)
        self.u.values = self.u.values.type(dtype).to(device)

        
        #self.calc_xdot = calc_xdot              # wrap the calculate xdot
        self.calc_xdot = Nlplant(device, dtype, 'C')

        # instantiate calc_xdot_mpc, this could be functional, but in the interest of not passing
        # about a huge amount of functions every time I have made it a classed.
        self.calc_xdot_mpc = Calc_xdot_mpc(
                self.x.values,                  # a full set of states
                self.u.values,                  # a full set of inputs
                self.x._mpc_x_idx,              # the indices of the MPC states inside the list of all states
                self.u._mpc_u_idx)              # the indices of the MPC inputs inside the list of all inputs

        # device, dtype redundant here for now as still uses scipy minimize - should rework for pytorch
        self.trim = Trim(device, dtype)                        # wrap the trim function
       
        # instantiate get observation classes, sure making these classes is definitely overkill
        # but it does allow for separation of modules and needs less passing around of variables
        # also this is a very slow way of doing things, if you can think of a better way please implement
        self.get_obs = Get_observation(self.x._obs_x_idx)
        self.get_obs_mpc = Get_observation(self.x._mpc_obs_x_idx)
        
        # instantiate the lineariser classes, again this could be functional, or a method of this class
        # but in keeping with UNIX philosphy I am keeping things modular and separate
        self.linmod_mpc = Linmod(
                self.calc_xdot_mpc,             # the input output nonlinear plant
                self.get_obs_mpc,               # takes x.values, u.values as input 
                len(self.x._get_mpc_x()),       # the number of states
                len(self.u._get_mpc_u()),       # the number of inputs
                len(self.x._mpc_obs_x_idx),     # the number of observable states
                1e-05,                          # the perturbation for linearisation
                self.paras.dt)                  # the simulation timestep (assumed to be constant)

        self.linmod_std = Linmod(
                self.calc_xdot,                 # the input output nonlinear plant
                self.get_obs,                   # the func to generate observable states
                len(self.x.values),             # the number of states
                len(self.u.values),             # the number of inputs
                len(self.x._obs_x_idx),         # the number of observable states
                1e-05,                          # the perturbation for linearisation
                self.paras.dt)                  # the simulation timestep (assumed to be constant)

        # discrete time LQR gain calculator
        self.dlqr = dlqr

    def step(self, u):
        """
        Function to update the state based on an instantaneous input
        """
        xdot = self.calc_xdot(self.x.values, u)[0]
        dx = xdot*self.paras.dt
        self.x.values += dx
       
    def calc_SFB_u(self, x, u):
        """
        Function to perform state feedback (SFB) to calculate an input (u) for the plant.
        This is for applying DLQR primarily.
        """
        
