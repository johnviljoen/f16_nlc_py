from control.trim import trim
import torch
import gym
import numpy as np
from scipy.signal import cont2discrete

from dynamics.parameters import state_vector, input_vector, simulation_parameters
from dynamics.nlplant import calc_xdot
from control.trim import trim
from control.lqr import dlqr
from utils.linmod import Linmod

class F16(gym.Env):
    
    def __init__(self):
        
        super().__init__()
        
        self.x = state_vector                   # mutable state dataclass
        self.u = input_vector                   # mutable input dataclass
        self.paras = simulation_parameters      # immutable simulation parameters dataclass
        
        self.calc_xdot = calc_xdot              # wrap the calculate xdot in this object
        self.trim = trim                        # wrap the trim function
        
        # instantiate the lineariser
        self.linmod = Linmod(
                self.calc_xdot_mpc,                 # the input output nonlinear plant
                self.get_obs_mpc,               # the func to generate observable states
                len(self.x._get_mpc_x()),             # the number of states
                len(self.u._get_mpc_u()),             # the number of inputs
                len(self.x._mpc_obs_x_idx),         # the number of observable states
                1e-05,                          # the perturbation for linearisation
                self.paras.dt)                  # the simulation timestep (assumed to be constant)
        
        self.dlqr = dlqr

    def get_obs(self, x, u):
        return torch.tensor([x[i] for i in self.x._obs_x_idx])

    def get_obs_mpc(self, x, u):
        return torch.tensor([x[i] for i in self.x._mpc_obs_x_idx])

    def calc_xdot_mpc(self, x, u):
        """
        Args:
            x:
                {h,phi,theta,V,alpha,beta,p,q,r,lf1,lf2}

            u:
                torch 2D tensor (vertical vector) of 3 elements
                {dh,da,dr}

        Returns:
            xdot:
                torch 2D tensor (vertical vector) of 10 elements
                time derivatives of {h,phi,theta,alpha,beta,p,q,r,lf1,lf2}
        """

        # without assertions we just get a seg fault, this is much easier to debug
        assert len(x) == 9, f"ERROR: expected 9 states, got {len(x)}"  
        assert len(u) == 3, f"ERROR: expected 3 inputs, got {len(u)}"

        # take the current full state as the starting point, and add the mpc states 
        std_x = torch.clone(self.x.values)
        mpc_x = x #self.x._get_mpc_x()
        for mpc_i, std_i in enumerate(self.x._mpc_x_idx):
            std_x[std_i] = mpc_x[mpc_i]
       
        std_u = torch.clone(self.u.values)
        mpc_u = u
        for mpc_i, std_i in enumerate(self.u._mpc_u_idx):
            std_u[std_i] = mpc_u[mpc_i]

        std_xdot = self.calc_xdot(std_x, std_u)[0]
        mpc_xdot = torch.zeros(len(mpc_x))
        for mpc_i, std_i in enumerate(self.x._mpc_x_idx):
            mpc_xdot[mpc_i] = std_xdot[std_i]
        
        return mpc_xdot

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
        
