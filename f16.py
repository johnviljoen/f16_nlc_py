from control.trim import trim
import torch
import gym
import numpy as np
from scipy.signal import cont2discrete

from dynamics.parameters import state_vector, input_vector, simulation_parameters
from dynamics.nlplant import calc_xdot
from control.trim import trim

class F16(gym.Env):
    
    def __init__(self):
        
        super().__init__()
        
        self.x = state_vector                   # mutable state dataclass
        self.u = input_vector                   # mutable input dataclass
        self.paras = simulation_parameters      # immutable simulation parameters dataclass
        
        self.calc_xdot = calc_xdot              # wrap the calculate xdot in this object
        self.trim = trim                        # wrap the trim function
        
        # instantiate some tensors to speed up linearisation process
        self.A = torch.zeros([len(self.x.values),len(self.x.values)])
        self.B = torch.zeros([len(self.x.values),len(self.u.values)])
        self.C = torch.zeros([len(self.x._obs_x_idx), len(self.x.values)])
        self.D = torch.zeros([len(self.x._obs_x_idx), len(self.u.values)])
        self.eps = 1e-05

    def linmod(self, x, u):

        # Perturb each of the state variables and compute linearisation
        for i in range(len(x)):

            dx = torch.zeros([len(x)])
            dx[i] = self.eps

            self.A[:,i] = (self.calc_xdot(x + dx, u)[0] - self.calc_xdot(x, u)[0]) / self.eps
            self.C[:,i] = (self.get_obs(x + dx, u)[0] - self.get_obs(x, u)[0]) / self.eps

        for i in range(len(u)):

            du = torch.zeros([len(u)])
            du[i] = self.eps

            self.B[:,i] = (self.calc_xdot(x, u + du)[0] - self.calc_xdot(x, u)[0]) / self.eps
            self.D[:,i] = (self.get_obs(x, u + du)[0] - self.get_obs(x, u)[0]) / self.eps 

        self.A, self.B, self.C, self.D = cont2discrete((self.A, self.B, self.C, self.D), self.paras.dt)[0:4]
   
    def get_obs(self, x, u):
        return torch.tensor([x[i] for i in self.x._obs_x_idx])

    def step(self, u):
        """
        Function to update the state based on an instantaneous input
        """
        xdot = self.calc_xdot(self.x.values, u)[0]
        dx = xdot*self.paras.dt
        self.x.values += dx
       
        

        # print(self.calc_xdot(self.x.values))
        
        # trim and linearise upon initialisation
        #self.x.initial_condition, _ = self.trim(10000, 700)
        #self.u.initial_condition = np.copy(self.x.initial_condition[12:16])
        # self.reset()
        
        # self.action_space = spaces.Box(low=self.u.lower_cmd_bound, high=self.u.upper_cmd_bound, dtype=np.float32)
        # self.observation_space = spaces.Box(low=self.x.lower_bound, high=self.x.upper_bound, shape=(len(self.x.states)), dtype=np.float32)
