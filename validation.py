#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:55:15 2021

@author: johnviljoen
"""

import unittest
import tqdm
import numpy as np

from env_c import F16
from utils import *
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

from parameters import x_lb, x_ub, u_lb, u_ub, udot_lb, udot_ub, SS

class test_F16(unittest.TestCase, F16):
    
    def __init__(self, state_vector, input_vector, simulation_parameters, nlplant):
        super(unittest.TestCase, self).__init__(state_vector, input_vector, simulation_parameters, nlplant)
    
    def LQR(self, linear=False, alpha_dist=1, beta_dist=1):
        
        # setup LQR controlled simulation
        self.paras.time_end = 10
        
        rng = np.linspace(self.paras.time_start, self.paras.time_end, int((self.paras.time_end-self.paras.time_start)/self.paras.dt))
        
        # create storage
        x_storage = np.zeros([len(rng),len(self.x.values)])
        u_storage = np.zeros([len(rng),len(self.u.values)])
        
        # linearise system
        A,B,C,D = self.linearise(self.x._get_mpc_x(), self.u._get_mpc_u(), calc_xdot=self.calc_xdot_na, get_obs=self.get_obs_na)
        
        # find the state weighting matrix
        Q = C.T@C
        
        # only care about p,q,r
        Q[0,0] = 0
        Q[1,1] = 0
        Q[2,2] = 0
        Q[3,3] = 0
        Q[7,7] = 0
        Q[8,8] = 0
        
        # find the input weighting matrix
        R = np.eye(3)
        
        # compute the DARE for LQR K
        K = dlqr(A, B, Q, R)
        
        if linear: # simulating on the linearised state space system
        
            x = np.zeros(len(self.x.mpc_states))
            x_eq = np.copy(x)
            u0 = np.zeros(len(self.u.mpc_inputs))
            u = np.copy(u0)
            
            # create storage
            x_storage = np.zeros([len(rng),len(x)])
            u_storage = np.zeros([len(rng),len(u)])
            
            # create zero reference for state vector
            x_ref = np.zeros(len(self.x.mpc_states))
            
            # apply disturbance to alpha, beta
            x[2] = alpha_dist * np.pi/180
            x[3] = beta_dist * np.pi/180
            
            # initial state
            u0 = np.zeros(len(self.u.mpc_inputs))
                        
            u_eq = self.u._get_mpc_u()
            x_eq = np.copy(x)

            for idx, val in enumerate(rng):
                
                print('idx:', idx)
                
                u = - K @ (x - x_ref)
                                                                                
                x = A @ x + B @ u 
                                
                x_storage[idx,:] = x + x_eq # cheat a bit here lmao
                u_storage[idx,:] = u + u_eq # cheat a bit here lmao
                
            vis_mpc_u(u_storage, rng)
            vis_mpc_x(x_storage, rng)
        
        else: # as you might expect... simulating on the nonlinear system
        
            x = self.x._get_mpc_x()
            u0 = self.u.initial_condition[1:]
            u = np.copy(u0)
            
            p_dem = 0
            q_dem = 0
            r_dem = 0
            
            for idx, val in enumerate(rng):
            
                print('idx:', idx)
                
                # change the current state reference to be the same as the current
                # state with the exception of the states we are controlling. This
                # works in nonlinear, but not linear, it is very unintuitive.
                x_ref = np.copy(x)
                x_ref[4] = p_dem
                x_ref[5] = q_dem
                x_ref[6] = r_dem
                
                # use the built in function for determining LQR action for consistency
                u = - K @ (x - x_ref) + u0
                
                # of course we arent controlling T here so only change final 3 inputs
                self.u.values[1:] = u                
                
                print('u:',self.u.values)
                
                # use the u values calculated to use the built in step function
                self.step(self.u.values)
                x = self.x._get_mpc_x()
                
                x_storage[idx,:] = self.x.values
                
            vis_x(x_storage, rng)
            vis_u(u_storage, rng)
            
            
    
    # def MPC(self, linear=False, unconstrained=False):
        
    #     # setup LQR controlled simulation
    #     self.paras.time_end = 1
        
    #     rng = np.linspace(self.paras.time_start, self.paras.time_end, int((self.paras.time_end-self.paras.time_start)/self.paras.dt))
        
    #     # create storage
    #     x_storage = np.zeros([len(rng),len(self.x.values)])
    #     u_storage = np.zeros([len(rng),len(self.u.values)])
        
    #     # find the x_lin, u_lin
    #     x_lin = self.x._get_mpc_x()
    #     u_lin = self.u._get_mpc_u()
        
    #     # linearise system
    #     A,B,C,D = self.linearise(x_lin, u_lin, calc_xdot=self.calc_xdot_na, get_obs=self.get_obs_na)
        
    #     for idx, val in enumerate(rng):
            
    #         print('idx:', idx)
    #         print('u:',self.u.values)
        
    #         # perform OSQP opt
    #         du = self.calc_MPC_action(0,0.01,0,A,B,C,D, x_lin, u_lin, 10, unconstrained=unconstrained)
            
    #         # du = results.x[0:3]
            
    #         print(du.shape)
            
    #         self.u.values[1:] += du[0:3,0]
        
    #         # use the u values calculated to use the built in step function
    #         self.step(self.u.values)
        
    #         x = self.x._get_mpc_x()
            
    #         x_storage[idx,:] = self.x.values
            
    #     vis_x(x_storage, rng)
    #     vis_u(u_storage, rng)
        
        
    def Example(self, notes=True, hzn=4, samples=10, TGM=True, unconstrained=True, return_MC=False):
        
        if notes:
            
            A = np.array([[1.1,2],[0,0.95]])
            B = np.array([[0],[0.0787]])
            C = np.array([[-1, 1]])
            x0 = np.array([[0.5, -0.5]]).T
            
        else:
            
            self.paras.dt = 0.01
            A,B,C,D = self.linearise(self.x._get_mpc_x(), self.u._get_mpc_u(), calc_xdot=self.calc_xdot_na, get_obs=self.get_obs_na)
            x = self.x._get_mpc_x()[:,None]
            x_ref = np.copy(x)
            x_ref[4] = 0
            x_ref[5] = 0
            x_ref[6] = 0
            x0 = x - x_ref
            x0 = self.x._get_mpc_x()[:,None]
        
        m = A.shape[0]
        n = B.shape[1]
        
        MM, CC = calc_MC(A, B, hzn)
        
        if return_MC:
            
            return MM, CC
        
        Q = C.T @ C
        R = np.eye(n) * 0.01
        
        QQ, RR = calc_QR(B.shape[1], C, hzn)
        
        ### not done on first part of notes and SLIGHTLY alters the outcome L ###
        ### despite the notes claiming them to be identical, this is the only ###
        ### discrepancy I have found between LQR and unconstrained MPC ###
        
        if TGM:
        
            LQR_K = - dlqr(A, B, Q, R)
            
            Q_bar = scipy.linalg.solve_discrete_lyapunov((A + B @ LQR_K).T, Q + LQR_K.T @ R @ LQR_K)
            
            QQ[-m:,-m:] = Q_bar
        
        ### upon further inspection it IMPROVES predictions - its dead on ###
        ### end of alteration ###
        
        H, F, G = calc_HFG(A, B, QQ, RR)
        
        L = calc_L(H,F,B.shape[1],hzn)
        
        u_opt1 = L @ x0
        
        
        # import pdb
        # pdb.set_trace()
        
                
        ex_CC = np.array([
                        [0,	        0,	         0,	        0],
                        [0.0787,	    0,	         0,	        0],
                        [0.1574,	    0,	         0,	        0],
                        [0.074765,  0.0787,	     0,	        0],
                        [0.32267, 	0.1574,	     0,	        0],
                        [0.0710267,	0.074765	,    0.0787,   	0],
                        [0.496991,	0.32267,	     0.1574,    	0],
                        [0.0674754,	0.0710267,	 0.074765,	0.0787]])
        
        ex_MM = np.array([
                        [1.1, 	    2],
                        [0,     	    0.95],
                        [1.21,	    4.1],
                        [0,	        0.9025],
                        [1.331,	    6.315],
                        [0,	        0.857375],
                        [1.4641,	    8.66125],
                        [0,	        0.814506]])
        
        ex_H = np.array([
                        [0.27083,	0.122376,	0.0156887,	-0.0338028],
                        [0.122376,  0.0863466,	0.0142912,	-0.0198043],
                        [0.0156887,	0.0142912,	0.0230222,	-0.00650337],
                        [-0.0338028,	-0.0198043,	-0.00650337,	0.0161937]])
        
        ex_F = np.array([
                        [0.977209,	4.92526],
                        [0.383191,	2.17393],
                        [0.0162362,	0.218901],
                        [-0.115225,	-0.617539]])
        
        ex_L = np.array([[-4.35631,	-18.6889]])
        
        ex_Q_bar = np.array([
            [3.91525	, 4.82686],
            [4.82686,	13.8564]])
        
        # if N == 4:

        #     np.testing.assert_almost_equal(CC, ex_CC, decimal=6)
        #     np.testing.assert_almost_equal(MM, ex_MM, decimal=6)
        #     np.testing.assert_almost_equal(H, ex_H, decimal=6)
        #     np.testing.assert_almost_equal(F, ex_F, decimal=5)
        #     np.testing.assert_almost_equal(L, ex_L, decimal=5)
        #     np.testing.assert_almost_equal(Q_bar, ex_Q_bar, decimal=4)
                        
        x = np.copy(x0)
        u = np.copy(u_opt1)
        u_storage = np.zeros([samples+1,n])
        u_storage[0,:] = np.squeeze(u)
        x_storage = np.zeros([samples+1,m])
        x_storage[0,:] = np.squeeze(x[:,0])
        y_storage = np.zeros([samples+1,m])
        
        y_storage[0,:] = np.squeeze(C @ x)
        
        def MPC_pred(C,x,H,F,MM,CC,hzn):
            
            u_pred = - np.linalg.inv(H) @ F @ x
            x_pred = MM @ x + CC @ u_pred
            
            # y_pred = np.tile(C, (1,hzn)) @ x_pred
            y_pred = np.zeros([hzn+1, np.max((C @ x).shape)])
            y_pred[0,:] = np.squeeze(C @ x)
            
            return u_pred, x_pred, y_pred
        
        u_pred, x_pred, y_pred = MPC_pred(C,x,H,F,MM,CC,hzn) 
        
        for i in range(hzn):
            
            y_pred[i+1,:] = np.squeeze(C @ x_pred[B.shape[0] * i:B.shape[0] * (i+1)])

        for i in range(samples):
            
            x = A @ x + B @ u
            
            u = L @ x
            
            # print(u_storage.shape)
            # print(u.shape)
            
            u_storage[i+1,:] = u[:,0]
            
            x_storage[i+1,:] = x[:,0]
            
            y_storage[i+1] = np.squeeze(C @ x)
            
        fig, axs = plt.subplots(2,1)
        
        axs[0].step(range(samples+1), u_storage, where='post')
        
        # print(np.append(u_opt1,u_pred))
        # print( np.reshape(np.append(u_opt1,u_pred),[hzn+1,n],order='C'))
        
        u_pred = np.reshape(np.append(u_opt1,u_pred),[hzn+1,n],order='C')
        
        axs[0].step(range(hzn+1), u_pred, 'g--', where='pre')
        axs[0].set_ylabel('u')
        
        axs[1].plot(range(samples+1), y_storage)
        axs[1].plot(range(hzn+1), y_pred, 'g--')
        axs[1].set_ylabel('y')
        axs[1].set_xlabel('sample')
        
        return L, u_pred, x_pred, y_pred
    
    def MPC(self):
        
        hzn = 10
        samples = 100
        self.paras.time_end = 5
        
        x_lin = self.x._get_mpc_x()
        u_lin = self.u._get_mpc_u()
        
        A,B,C,D = self.linearise(x_lin, u_lin, calc_xdot=self.calc_xdot_na, get_obs=self.get_obs_na)
        ss = SS(A,B,C,D,x_lin,u_lin,self.paras.dt)
        x0 = self.x._get_mpc_x() - self.x._get_mpc_x()
        
        rng = np.linspace(self.paras.time_start, self.paras.time_end, int((self.paras.time_end-self.paras.time_start)/self.paras.dt))
        
        # create storage
        x_storage = np.zeros([len(self.x.values), len(rng)])
        u_storage = np.zeros([len(self.u.values), len(rng)])
        
        p_dem = 0
        q_dem = 0
        r_dem = 0
        
        x = np.copy(x_lin)
        u0 = np.copy(u_lin)
        u = np.copy(u0)
        x_ref = np.copy(x)
        
        dx = x - x_ref
        
        u_opt1, u_pred, x_pred, y_pred = self.calc_MPC_action(ss, dx, hzn, constrain=True, return_pred=True, polish=False, warm_start=True)  
        
        
        self.x.values[8] = 0.05
        x = self.x._get_mpc_x()
        # x[2] = 0.05
        
        for idx, val in tqdm.tqdm(enumerate(rng)):
            
            
            # x_ref = np.copy(x)
            x_ref[4] = p_dem
            x_ref[5] = q_dem
            x_ref[6] = r_dem
        
            dx = x - x_ref
            
            u = self.calc_MPC_action(ss, dx, hzn, constrain=True, return_pred=False, polish=False, warm_start=True)  
                        
            self.u.values[1:] = np.squeeze(u) + u_lin 
            
            # use the u values calculated to use the built in step function
            self.step(self.u.values)
            
            x = self.x._get_mpc_x()
            
            x_storage[:,idx] = self.x.values
            
        fig, axs = plt.subplots(3,4)
        
        # import pdb
        # pdb.set_trace()
        
        axs[0][0].step(rng, x_storage[0,:], 'blue', where='pre')
        axs[0][0].set_ylabel('npos')
        
        axs[1][0].step(rng, x_storage[1,:], 'blue', where='pre')
        axs[1][0].set_ylabel('epos')
        
        axs[2][0].step(rng, x_storage[2,:], 'blue', where='pre')
        axs[2][0].set_ylabel('h')
        
        axs[0][1].step(rng, x_storage[3,:], 'blue', where='pre')
        axs[0][1].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,0], 'red', where='pre')
        axs[0][1].set_ylabel('phi')
        
        axs[1][1].step(rng, x_storage[4,:], 'blue', where='pre')
        axs[1][1].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,1], 'red', where='pre')
        axs[1][1].set_ylabel('theta')
                             
        axs[2][1].step(rng, x_storage[5,:], 'blue', where='pre')
        axs[2][1].set_ylabel('psi')
        
        axs[0][2].step(rng, x_storage[6,:], 'blue', where='pre')
        axs[0][2].set_ylabel('V')
        
        axs[1][2].step(rng, x_storage[7,:], 'blue', where='pre')
        axs[1][2].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,2], 'red', where='pre')
        axs[1][2].set_ylabel('alpha')
                             
        axs[2][2].step(rng, x_storage[8,:], 'blue', where='pre')
        axs[2][2].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,3], 'red', where='pre')
        axs[2][2].set_ylabel('beta')
        
        axs[0][3].step(rng, x_storage[9,:], 'blue', where='pre')
        axs[0][3].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,4], 'red', where='pre')
        axs[0][3].set_ylabel('p')
        
        axs[1][3].step(rng, x_storage[10,:], 'blue', where='pre')
        axs[1][3].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,5], 'red', where='pre')
        axs[1][3].set_ylabel('q')
                             
        axs[2][3].step(rng, x_storage[11,:], 'blue', where='pre')
        axs[2][3].step(np.linspace(0,self.paras.dt * hzn,hzn+1), y_pred[:,6], 'red', where='pre')
        axs[2][3].set_ylabel('r')
        
        # vis_x(x_storage, rng)
        # vis_u(u_storage, rng)
        
    # def Example_on_F16(self, A, B, C, D, x0, hzn):
        
        