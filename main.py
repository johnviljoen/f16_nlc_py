#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:23:36 2021

@author: johnviljoen
"""

# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_checker import check_env
from sys import exit
import torch
import matplotlib.pyplot as plt

# custom files
from f16 import F16

def main():
    # run through tests or nah
    f16 = F16()
    #print(f16.calc_xdot(f16.x.values, f16.u.values))
    f16.step(f16.u.values) 
    #print(f16.trim(1000,700,f16.x, f16.u))

    """ Run simulation for 0.5 seconds """
    # number of timesteps
    ts = 4000
    out = torch.zeros([ts,18])
    for i in range(ts):
        print(i)
        f16.step(f16.u.values)
        out[i,:] = f16.x.values

    t = torch.linspace(0,0.5,ts)
    fig, axs = plt.subplots(18,1)
    for i in range(18):
        axs[i].plot(t, out[:,i])
    plt.show()

    import pdb
    pdb.set_trace()

    # instantiate base class and testing class
    #f16 = F16(state_vector, input_vector, simulation_parameters, nlplant)
    #test_f16 = test_F16(state_vector, input_vector, simulation_parameters, nlplant)

    #x_lin = f16.x._get_mpc_x()
    #u_lin = f16.u._get_mpc_u()

    #A,B,C,D = f16.linearise(x_lin, u_lin, calc_xdot=f16.calc_xdot_na, get_obs=f16.get_obs_na)

    #if testing:
        # u_opt = f16.calc_MPC_action(0.01,0,0,A,B,C,D, x_lin, u_lin, 10, unconstrained=True)
        # test_f16.MPC(unconstrained=True)
        # test_f16.LQR(linear=True, alpha_dist=1, beta_dist=0)
        # test_f16.LQR(linear=False)
        # test_f16.Example(notes=False, hzn=100, samples=1000, TGM=True)
        # test_f16.Example(notes=True, hzn=4, samples=10, TGM=True, return_MC=False)
        # test_f16.unconstrained_MPC_nl()
        
        #u_seq = np.tile(u_lin[:,None],3)
        
        #ss = SS(A,B,C,D,x_lin,u_lin,f16.paras.dt)
        #x0 = f16.x._get_mpc_x() - f16.x._get_mpc_x()
        #hzn = 3
        #u_opt1, u_pred, x_pred, y_pred = f16.calc_MPC_action(ss, x0, hzn, constrain=True, return_pred=True, polish=False, warm_start=True)
        #import pdb
        #pdb.set_trace() 
        #test_f16.MPC()


if __name__ == "__main__":
    main()



