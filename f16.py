import torch
import gym
import numpy as np

from dynamics.parameters import state_vector, input_vector, simulation_parameters, nlplant
from dynamics.nlplant import calc_xdot

class F16(gym.Env):
    
    def __init__(self):
        
        super().__init__()
        
        self.x = state_vector                   # mutable state dataclass
        self.u = input_vector                   # mutable input dataclass
        self.paras = simulation_parameters      # immutable simulation parameters dataclass
        self.nlplant = nlplant                  # C interface - the heart of the simulation

        self.calc_xdot = calc_xdot

        print(self.calc_xdot(self.x.values))
        
        # trim and linearise upon initialisation
        #self.x.initial_condition, _ = self.trim(10000, 700)
        #self.u.initial_condition = np.copy(self.x.initial_condition[12:16])
        # self.reset()
        
        # self.action_space = spaces.Box(low=self.u.lower_cmd_bound, high=self.u.upper_cmd_bound, dtype=np.float32)
        # self.observation_space = spaces.Box(low=self.x.lower_bound, high=self.x.upper_bound, shape=(len(self.x.states)), dtype=np.float32)