#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:18:13 2021

@author: johnviljoen
"""

# dependencies
import numpy as np
from numpy import pi
import gym
from gym import spaces
from scipy.optimize import minimize
import ctypes
import os
import tqdm
from scipy.signal import cont2discrete
import scipy
import osqp
from scipy.sparse import csc_matrix
from sys import exit
from dynamics.parameters import SS
from dynamics import actuators

# custom files
from utils import *

class F16(gym.Env):
    
    def __init__(self, state_vector, input_vector, simulation_parameters, nlplant):
        
        super().__init__()
        
        self.x = state_vector                   # mutable state dataclass
        self.u = input_vector                   # mutable input dataclass
        self.paras = simulation_parameters      # immutable simulation parameters dataclass
        self.nlplant = nlplant                  # C interface - the heart of the simulation

        self.calc_xdot(self.x.values, self.u.values)
        
        # trim and linearise upon initialisation
        self.x.initial_condition, _ = self.trim(10000, 700)
        self.u.initial_condition = np.copy(self.x.initial_condition[12:16])
        # self.reset()
        
        # self.action_space = spaces.Box(low=self.u.lower_cmd_bound, high=self.u.upper_cmd_bound, dtype=np.float32)
        # self.observation_space = spaces.Box(low=self.x.lower_bound, high=self.x.upper_bound, shape=(len(self.x.states)), dtype=np.float32)
        
    def calc_xdot(self, x, u):
        
        """ calculates, and returns the rate of change of the state vector, x, using the empirical
        aerodynamic data held in folder 'C', also using equations of motion found in the
        shared library C file. This function includes all actuator models.
        
        Args:
            x:
                numpy 2D array (vertical vector) of 18 elements
                {xe,ye,h,phi,theta,psi,V,alpha,beta,p,q,r,T,dh,da,dr,lf2,lf1}
            u:
                numpy 2D array (vertical vector) of 4 elements
                {T,dh,da,dr}
    
        Returns:
            xdot:
                numpy 2D array (vertical vector) of 18 elements
                time derivatives of {xe,ye,h,phi,theta,psi,V,alpha,beta,p,q,r,T,dh,da,dr,lf2,lf1}
        """
        
        # initialise variables
        xdot = np.zeros(18)
        temp = np.zeros(6)
        coeff = np.zeros(3)
        # Thrust Model
        temp[0] = actuators.upd_thrust(u[0], x[12])
        # Dstab Model
        temp[1] = actuators.upd_dstab(u[1], x[13])
        # aileron model
        temp[2] = actuators.upd_ail(u[2], x[14])
        # rudder model
        temp[3] = actuators.upd_rud(u[3], x[15])
        # leading edge flap model
        temp[5], temp[4] = actuators.upd_lef(x[2], x[6], coeff, x[7], x[17], x[16], self.nlplant)
        # run nlplant for xdot
        import pdb
        pdb.set_trace()
        self.nlplant.Nlplant(ctypes.c_void_p(x[:17].ctypes.data), ctypes.c_void_p(xdot.ctypes.data), ctypes.c_int(self.paras.fi_flag))    
        # assign actuator xdots
        xdot[12:18] = temp
        return xdot
    
    def step(self, action):
        
        """
        action of form 1D numpy array:
            Thrust demand
            dh
            da
            dr
        """
        
        # check the current state isnt outside of C lookup table bounds
        
        bounds_check = [self.x.values[i] < self.x.lower_bound[i] or self.x.values[i] > self.x.upper_bound[i] for i in range(len(self.x.values))]
        
        # if any bounds check return the F16 has left envelope, break and raise exception
        # https://stackoverflow.com/questions/52069575/check-if-numpy-array-falls-within-bounds - a better way to do this
        if any(bounds_check):
            
            print('A state has left the flight envelope designated by the lookup tables')
            exit()
        
        self.x.values += self.calc_xdot(self.x.values, action)*self.paras.dt
        reward = 1
        isdone = False
        info = {'fidelity':'high'}
        return self.get_obs(self.x.values, self.u.values), reward, isdone, info
    
    def reset(self):
        self.x.values = np.copy(self.x.initial_condition)
        self.u.values = np.copy(self.u.initial_condition)
        return self.get_obs(self.x.values, self.u.values)
    
    def get_obs(self, x, u):
        
        """ Function for acquiring the current observation from the state space.
        
        Args:
            x -> the state vector
            of form numpy 2D array (vertical vector)
            
        Returns:
            y -> system output
            of form numpy 1D array to match gym requirements
        """
        
        return np.array([x[i] for i in self.x._obs_x_idx])
    
    def calc_xdot_na(self, x, u):
        """
        Args:
            x:
                {h,phi,theta,V,alpha,beta,p,q,r,lf1,lf2}

            u:
                numpy 2D array (vertical vector) of 3 elements
                {dh,da,dr}
    
        Returns:
            xdot:
                numpy 2D array (vertical vector) of 10 elements
                time derivatives of {h,phi,theta,alpha,beta,p,q,r,lf1,lf2}
        """
        state_vector = np.copy(self.x.values)
        # np.zeros(18)
        
        input_vector = np.copy(self.u.values)
                
        for i in range(len(self.x._mpc_x_idx)):
            state_vector[self.x._mpc_x_idx[i]] = x[i]
            
        for i in range(len(self.u._mpc_u_idx)):
            state_vector[self.x._mpc_u_in_x_idx[i]] = u[i]
            input_vector[self.u._mpc_u_idx[i]] = u[i]
        
        # initialise variables
        xdot = np.zeros(18)
        coeff = np.zeros(3)
        C_input_x = np.zeros(18)
        
        # leading edge flap model
        lf_state1_dot, lf_state2_dot = upd_lef(state_vector[2], state_vector[6], coeff, state_vector[7], state_vector[17], state_vector[16], self.nlplant)
        # run nlplant for xdot
        # C_input_x = np.concatenate((x[0:12],u,x[13:14]))
        self.nlplant.Nlplant(ctypes.c_void_p(state_vector.ctypes.data), ctypes.c_void_p(xdot.ctypes.data), ctypes.c_int(self.paras.fi_flag))    
        # assign actuator xdots
        state_vector_dot = np.concatenate((xdot[0:12],np.zeros(4),np.array([lf_state1_dot, lf_state2_dot])))
                
        # return the xdot in the form of the original input vector it was input to
        # this function in.
        return np.array([state_vector_dot[i] for i in self.x._mpc_x_idx])
    
    def get_obs_na(self, x, u):
        return np.array([x[i] for i in self.x._mpc_obs_x_idx])
    
    def trim(self, h_t, v_t):
        
        """ Function for trimming the aircraft in straight and level flight. The objective
        function is built to be the same as that of the MATLAB version of the Nguyen 
        simulation.
        
        Args:
            h_t:
                altitude above sea level in ft, float
            v_t:
                airspeed in ft/s, float
                
        Returns:
            x_trim:
                trim state vector, 1D numpy array
            opt:
                scipy.optimize.minimize output information
        """
        
        def obj_func(UX0, h_t, v_t, fi_flag, nlplant):
    
            V = v_t
            h = h_t
            P3, dh, da, dr, alpha = UX0
            npos = 0
            epos = 0
            phi = 0
            psi = 0
            beta = 0
            p = 0
            q = 0
            r = 0
            rho0 = 2.377e-3
            tfac = 1 - 0.703e-5*h
            temp = 519*tfac
            if h >= 35000:
                temp = 390
            rho = rho0*tfac**4.14
            qbar = 0.5*rho*V**2
            ps = 1715*rho*temp
            dlef = 1.38*alpha*180/pi - 9.05*qbar/ps + 1.45
            x = np.array([npos, epos, h, phi, alpha, psi, V, alpha, beta, p, q, r, P3, dh, da, dr, dlef, -alpha*180/pi])
            
            # thrust limits
            x[12] = np.clip(x[12], self.u.lower_cmd_bound[0], self.u.upper_cmd_bound[0])
            # elevator limits
            x[13] = np.clip(x[13], self.u.lower_cmd_bound[1], self.u.upper_cmd_bound[1])
            # aileron limits
            x[14] = np.clip(x[14], self.u.lower_cmd_bound[2], self.u.upper_cmd_bound[2])
            # rudder limits
            x[15] = np.clip(x[15], self.u.lower_cmd_bound[3], self.u.upper_cmd_bound[3])
            # alpha limits
            x[7] = np.clip(x[7], self.x.lower_bound[7]*pi/180, self.x.upper_bound[7]*pi/180)
               
            u = np.array([x[12],x[13],x[14],x[15]])
            xdot = self.calc_xdot(x, u)
            
            phi_w = 10
            theta_w = 10
            psi_w = 10
            
            weight = np.array([0, 0, 5, phi_w, theta_w, psi_w, 2, 10, 10, 10, 10, 10]).transpose()
            cost = np.matmul(weight,xdot[0:12]**2)
            
            return cost
        
        # initial guesses
        thrust = 5000           # thrust, lbs
        elevator = -0.09        # elevator, degrees
        alpha = 8.49            # AOA, degrees
        rudder = -0.01          # rudder angle, degrees
        aileron = 0.01          # aileron, degrees
        
        UX0 = [thrust, elevator, alpha, rudder, aileron]
                
        opt = minimize(obj_func, UX0, args=((h_t, v_t, self.paras.fi_flag, self.nlplant)), method='Nelder-Mead',tol=1e-10,options={'maxiter':5e+04})
        
        P3_t, dstab_t, da_t, dr_t, alpha_t  = opt.x
        
        rho0 = 2.377e-3
        tfac = 1 - 0.703e-5*h_t
        
        temp = 519*tfac
        if h_t >= 35000:
            temp = 390
            
        rho = rho0*tfac**4.14
        qbar = 0.5*rho*v_t**2
        ps = 1715*rho*temp
        
        dlef = 1.38*alpha_t*180/pi - 9.05*qbar/ps + 1.45
        
        x_trim = np.array([0, 0, h_t, 0, alpha_t, 0, v_t, alpha_t, 0, 0, 0, 0, P3_t, dstab_t, da_t, dr_t, dlef, -alpha_t*180/pi])
        
        return x_trim, opt
        
    def linearise(self, x, u, calc_xdot=None, get_obs=None, discrete=True, stationary=True):
        
        """ Function to linearise the aircraft at a given state vector and input demand.
        This is done by perturbing each state and measuring its effect on every other state.
        
        Args:
            x:
                state vector, 2D numpy array (vertical vector)
            u:
                input vector, 2D numpy array (vertical vector)
                
        Returns:
            4 2D numpy arrays, representing the 4 discrete state space matrices, A,B,C,D.
        """
        
        if calc_xdot == None:
            calc_xdot = self.calc_xdot
            C = np.zeros([len(self.x._obs_x_idx),len(x)])
            D = np.zeros([len(self.x._obs_x_idx),len(u)])
        if get_obs == None:
            get_obs = self.get_obs
        if calc_xdot == self.calc_xdot_na:
            C = np.zeros([len(self.x._mpc_obs_x_idx),len(x)])
            D = np.zeros([len(self.x._mpc_obs_x_idx),len(u)])
        
        eps = 1e-05
        
        A = np.zeros([len(x),len(x)])
        B = np.zeros([len(x),len(u)])
        
        # Perturb each of the state variables and compute linearization
        for i in range(len(x)):
            
            dx = np.zeros([len(x)])
            dx[i] = eps
                        
            A[:, i] = (calc_xdot(x + dx, u) - calc_xdot(x, u)) / eps
            C[:, i] = (get_obs(x + dx, u) - get_obs(x, u)) / eps
            
        # Perturb each of the input variables and compute linearization
        for i in range(len(u)):
            
            du = np.zeros([len(u)])
            du[i] = eps
                    
            B[:, i] = (calc_xdot(x, u + du) - calc_xdot(x, u)) / eps
            D[:, i] = (get_obs(x, u + du) - get_obs(x, u)) / eps
            
        if discrete:
            
            return cont2discrete((A, B, C, D), self.paras.dt)[0:4]
        
        else:
            
            return A,B,C,D
        
    def MPC(self, ss, x0, hzn, TGM=True, unconstrained=True):
        
        """
        x0 is the dx, in validation it is x0 = x - x_ref
        """
        
        m = ss.A.shape[0]
        n = ss.B.shape[1]
        
        MM, CC = calc_MC(ss.A, ss.B, hzn)
        
        Q = ss.C.T @ ss.C
        R = np.eye(n) * 0.01
        
        QQ, RR = calc_QR(n, ss.C, hzn)
        
        if TGM:
        
            LQR_K = - dlqr(ss.A, ss.B, Q, R)
            
            Q_bar = scipy.linalg.solve_discrete_lyapunov((ss.A + ss.B @ LQR_K).T, Q + LQR_K.T @ R @ LQR_K)
            
            QQ[-m:,-m:] = Q_bar
            
        H, F, G = calc_HFG(ss.A, ss.B, QQ, RR)
        
        if unconstrained:
            
            L = calc_L(H,F,B.shape[1],hzn)
            
            u_opt1 = L @ x0
            
        else: 
            
            x0[2] = 0.1
            
            OSQP_P = 2 * H
            
            OSQP_q = (2 * x0.T @ F.T).T            
                        
            def calc_constr(x, x_lin, u_lin):
            
                # calculate state constraint limits vector relative to point of linearisation
                x_lb = np.array(self.x._mpc_x_lb)[:,None] + x_lin[:,None]
                x_ub = np.array(self.x._mpc_x_ub)[:,None] - x_lin[:,None]
                u_lb = np.array(self.u._mpc_u_lb)[:,None] + u_lin[:,None]
                u_ub = np.array(self.u._mpc_u_ub)[:,None] - u_lin[:,None]
                udot_lb = np.array(self.u._mpc_udot_lb)[:,None] * self.paras.dt # rates of change dont care about current state
                udot_ub = np.array(self.u._mpc_udot_ub)[:,None] * self.paras.dt # however they do care about the time step
                
                # remove one lf actuator limit -> caused primal infeasibility
                x_lb[-1] = -np.inf
                x_ub[-1] = np.inf
                
                x_lb = np.tile(x_lb,(hzn,1))    
                x_ub = np.tile(x_ub,(hzn,1))
                
                state_constr_lower = x_lb - MM @ x[:,None]
                state_constr_upper = x_ub - MM @ x[:,None]
                
                # the state constraint input sequence matrix is just CC
                
                # calculate the command saturation limits vector
                cmd_constr_lower = np.tile(u_lb,(hzn,1))
                cmd_constr_upper = np.tile(u_ub,(hzn,1))
                
                # calculate the command saturation input sequence matrix -> just eye
                cmd_constr_mat = np.eye(n*hzn)
                
                # calculate the command rate saturation limits vector
                # needs to be relative to the linearisation point (zero)
                u0_rate_constr_lower = 0 + udot_lb #* self.paras.dt
                u0_rate_constr_upper = 0 + udot_ub #* self.paras.dt
                
                cmd_rate_constr_lower = np.concatenate((u0_rate_constr_lower,np.tile(udot_lb,(hzn-1,1))))
                cmd_rate_constr_upper = np.concatenate((u0_rate_constr_upper,np.tile(udot_ub,(hzn-1,1))))
                
                # calculate the command rate saturation input sequence matrix
                cmd_rate_constr_mat = np.eye(n*hzn)
                for i in range(n*hzn):
                    if i >= n:
                        cmd_rate_constr_mat[i,i-n] = -1
                        
                # assemble the complete matrices to send to OSQP
                OSQP_A = np.concatenate((CC, cmd_constr_mat, cmd_rate_constr_mat), axis=0)
                OSQP_l = np.concatenate((state_constr_lower, cmd_constr_lower, cmd_rate_constr_lower))
                OSQP_u = np.concatenate((state_constr_upper, cmd_constr_upper, cmd_rate_constr_upper))
                
                # unconstrain
                # OSQP_l += -np.inf
                # OSQP_u += np.inf
                return OSQP_A, OSQP_l, OSQP_u
            
            OSQP_A, OSQP_l, OSQP_u = calc_constr(x0, ss.x_lin, ss.u_lin)
            
            OSQP_P = csc_matrix(OSQP_P)
            OSQP_A = csc_matrix(OSQP_A)            
            
            solver = osqp.OSQP()
            solver.setup(P=OSQP_P, q=OSQP_q, A=OSQP_A, l=OSQP_l, u=OSQP_u, polish=True, warm_start=True)
            
            results = solver.solve()
            
            # u_seq = np.load('u_seq.npy')
            
            
         
        
        # setup simulation
        u_opt1 = results.x[0:n][:,None]
        samples = 100
        x = np.copy(x0)[:,None]
        u = np.copy(u_opt1)
        u_storage = np.zeros([samples+1,n])
        u_storage[0,:] = np.squeeze(u)
        x_storage = np.zeros([samples+1,m])
        x_storage[0,:] = np.squeeze(x[:,0])
        y_storage = np.zeros([samples+1,m])
        y_storage[0,:] = np.squeeze(ss.C @ x)
        
        
        import pdb
        pdb.set_trace()   
        
        def MPC_pred(C,x,H,F,MM,CC,hzn):
            
            u_pred = - np.linalg.inv(H) @ F @ x
            x_pred = MM @ x + CC @ u_pred
            
            # y_pred = np.tile(C, (1,hzn)) @ x_pred
            y_pred = np.zeros([hzn+1, np.max((C @ x).shape)])
            y_pred[0,:] = np.squeeze(C @ x)
            
            return u_pred, x_pred, y_pred
        
        u_pred, x_pred, y_pred = MPC_pred(ss.C,x,H,F,MM,CC,hzn) 
        
        for i in range(hzn):
            
            y_pred[i+1,:] = np.squeeze(ss.C @ x_pred[m * i:m * (i+1)])

        for i in range(samples):
            
            x = ss.A @ x + ss.B @ u
            
            u = L @ x
            
            # print(u_storage.shape)
            # print(u.shape)
            
            u_storage[i+1,:] = u[:,0]
            
            x_storage[i+1,:] = x[:,0]
            
            y_storage[i+1] = np.squeeze(ss.C @ x)
            
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
        
        #return L, u_pred, x_pred, y_pred
        
    def calc_MPC_action(self, ss, x0, hzn, constrain=True, return_pred=False, polish=False, warm_start=True):
        
        """
        x0 is the dx, in validation it is x0 = x - x_ref
        """
        
        m = ss.A.shape[0]
        n = ss.B.shape[1]
        
        MM, CC = calc_MC(ss.A, ss.B, hzn)
        
        Q = ss.C.T @ ss.C
        Q[0,0] = 0
        Q[1,1] = 0
        Q[2,2] = 0
        Q[3,3] = 0
        Q[7,7] = 0
        Q[8,8] = 0
        R = np.eye(n) * 0.01
        
        QQ, RR = calc_QR(n, ss.C, hzn)
                
        LQR_K = - dlqr(ss.A, ss.B, Q, R)
        
        Q_bar = scipy.linalg.solve_discrete_lyapunov((ss.A + ss.B @ LQR_K).T, Q + LQR_K.T @ R @ LQR_K)
        
        QQ[-m:,-m:] = Q_bar
            
        H, F, G = calc_HFG(ss.A, ss.B, QQ, RR)
                        
        # x0[2] = 0.1
        
        OSQP_P = 2 * H
        
        OSQP_q = (2 * x0.T @ F.T).T     
        
        if return_pred:
            
            u_pred = - np.linalg.inv(H) @ F @ x0
            x_pred = MM @ x0 + CC @ u_pred
            
            # y_pred = np.tile(C, (1,hzn)) @ x_pred
            y_pred = np.zeros([hzn+1, np.max((ss.C @ x0).shape)])
            y_pred[0,:] = np.squeeze(ss.C @ x0)
            
            for i in range(hzn):
                y_pred[i+1,:] = np.squeeze(ss.C @ x_pred[m * i:m * (i+1)])
        
        """### Validated to here ###"""
                
        # calculate state constraint limits vector relative to point of linearisation
        x_lb = np.array(self.x._mpc_x_lb)[:,None] + ss.x_lin[:,None]
        x_ub = np.array(self.x._mpc_x_ub)[:,None] - ss.x_lin[:,None]
        u_lb = np.array(self.u._mpc_u_lb)[:,None] + ss.u_lin[:,None]
        u_ub = np.array(self.u._mpc_u_ub)[:,None] - ss.u_lin[:,None]
        udot_lb = np.array(self.u._mpc_udot_lb)[:,None] * self.paras.dt # rates of change dont care about current state
        udot_ub = np.array(self.u._mpc_udot_ub)[:,None] * self.paras.dt # however they do care about the time step
        
        # remove one lf actuator limit -> caused primal infeasibility
        x_lb[-1] = -np.inf
        x_ub[-1] = np.inf
        
        x_lb = np.tile(x_lb,(hzn,1))    
        x_ub = np.tile(x_ub,(hzn,1))
        
        state_constr_lower = x_lb - MM @ x0[:,None]
        state_constr_upper = x_ub - MM @ x0[:,None]
        
        # the state constraint input sequence matrix is just CC
        
        # calculate the command saturation limits vector
        cmd_constr_lower = np.tile(u_lb,(hzn,1))
        cmd_constr_upper = np.tile(u_ub,(hzn,1))
        
        # calculate the command saturation input sequence matrix -> just eye
        cmd_constr_mat = np.eye(n*hzn)
        
        # calculate the command rate saturation limits vector
        # needs to be relative to the linearisation point (zero)
        u0_rate_constr_lower = 0 + udot_lb #* self.paras.dt
        u0_rate_constr_upper = 0 + udot_ub #* self.paras.dt
        
        cmd_rate_constr_lower = np.concatenate((u0_rate_constr_lower,np.tile(udot_lb,(hzn-1,1))))
        cmd_rate_constr_upper = np.concatenate((u0_rate_constr_upper,np.tile(udot_ub,(hzn-1,1))))
        
        # calculate the command rate saturation input sequence matrix
        cmd_rate_constr_mat = np.eye(n*hzn)
        for i in range(n*hzn):
            if i >= n:
                cmd_rate_constr_mat[i,i-n] = -1
                
        # assemble the complete matrices to send to OSQP
        OSQP_A = np.concatenate((CC, cmd_constr_mat, cmd_rate_constr_mat), axis=0)
        OSQP_l = np.concatenate((state_constr_lower, cmd_constr_lower, cmd_rate_constr_lower))
        OSQP_u = np.concatenate((state_constr_upper, cmd_constr_upper, cmd_rate_constr_upper))
            
        if constrain:
            pass
        else:
            # unconstrain
            OSQP_l += -np.inf
            OSQP_u += np.inf
                
        OSQP_P = csc_matrix(OSQP_P)
        OSQP_A = csc_matrix(OSQP_A)            
        
        solver = osqp.OSQP()
        solver.setup(P=OSQP_P, q=OSQP_q, A=OSQP_A, l=OSQP_l, u=OSQP_u, polish=polish, warm_start=warm_start, verbose=False)
        
        results = solver.solve()
        
        u_opt1 = results.x[0:n][:,None]
        
        if return_pred:
            return u_opt1, u_pred, x_pred, y_pred
        else:
            return u_opt1
        
        
    
    # def calc_MPC_action(self, p_dem, q_dem, r_dem, A, B, C, D, x_lin, u_lin, hzn, unconstrained=False):
        
    #     """
    #     Calculate the optimal next step based on a dual-mode MPC with LQR infinite
    #     horizon. The current state vector is used for linearisation and therefore LQR gain
    #     calculation, finite horizon prediction, and input sequence optimisation.
        
    #     args:
    #         - p_dem: demanded roll rate in degrees/s
    #         - q_dem: demanded pitch rate in degrees/s
    #         - r_dem: demanded yaw rate in degrees/s
            
    #         - A, B, C, D: discrete state space matrices.
    #     """
        
    #     # convert x linearised point into degrees
    #     x_lin = x_lin[:,None]
    #     x_lin[0] = x_lin[0] * 180/np.pi
    #     x_lin[1] = x_lin[1] * 180/np.pi
    #     x_lin[2] = x_lin[2] * 180/np.pi
    #     x_lin[3] = x_lin[3] * 180/np.pi
    #     x_lin[4] = x_lin[4] * 180/np.pi
    #     x_lin[5] = x_lin[5] * 180/np.pi
    #     x_lin[6] = x_lin[6] * 180/np.pi
        
    #     u_lin = u_lin[:,None] # already in degrees
        
    #     # calculate the difference between the demanded and current state vectors
    #     # which will be the x fed into the linear system.
    #     x = self.x._get_mpc_x()
    #     x_ref = np.copy(x)
    #     x_ref[4] = p_dem
    #     x_ref[5] = q_dem
    #     x_ref[6] = r_dem
        
    #     # the delta x which will be used in the linear system. Check sign!
    #     dx = (x - x_ref)[:,None]
    #     #dx = - x_lin + x[:,None]
        
    #     # get the current actuator states for limits
    #     act_states = self.x.values[13:16]
        
    #     # calculate the state weighting matrix
    #     Q = C.T @ C
        
    #     # only care about p,q,r
    #     Q[0,0] = 0
    #     Q[1,1] = 0
    #     Q[2,2] = 0
    #     Q[3,3] = 0
    #     Q[7,7] = 0
    #     Q[8,8] = 0
        
    #     # calculate the input weighting matrix
    #     R = np.eye(len(self.u.mpc_inputs))
        
    #     # compute the DARE for LQR K, check sign!
    #     K = - dlqr(A, B, Q, R)
        
    #     m = len(self.x.mpc_states)      # number of states
    #     n = len(self.u.mpc_inputs)      # number of inputs
        
    #     #dx = dx[:,None]                   # convert x to vertical vector
        
    #     # no need for an x_ref as it is zero for the state space system
        
    #     # calculate matrices for predictions (p16 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    #     MM, CC = calc_MC(A, B, hzn)
        
    #     # calculate terminal weighting matrix (p24 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    #     Q_bar = scipy.linalg.solve_discrete_lyapunov((A + B @ K).T, Q + K.T @ R @ K)
            
    #     # construct full QQ, RR (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    #     QQ = dmom(Q, hzn)
    #     QQ[-m:,-m:] = Q_bar
    #     RR = dmom(R, hzn)
        
    #     H = CC.T @ QQ @ CC + RR
    #     F = CC.T @ QQ @ MM
    #     G = MM.T @ QQ @ MM
        
    #     # construct objective function (2.3) (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    #     # and implement this in OSQP format
        
    #     OSQP_P = 2 * H
        
    #     # OSQP_q = -2 * ((x_ref - MM @ x).T @ QQ @ CC).T
    #     OSQP_q = (2 * dx.T @ F.T).T
        
    #     if unconstrained:
    #         u_opt = -np.linalg.inv(H) @ F @ dx
    #         return u_opt
        
    #     # NOT THIS from parameters import u_lb, u_ub, x_lb, x_ub, udot_lb, udot_ub
        
    #     # calculate state constraint limits vector relative to point of linearisation
    #     x_lb = np.array(self.x._mpc_x_lb)[:,None] + x_lin
    #     x_ub = np.array(self.x._mpc_x_ub)[:,None] - x_lin
    #     u_lb = np.array(self.u._mpc_u_lb)[:,None] + u_lin
    #     u_ub = np.array(self.u._mpc_u_ub)[:,None] - u_lin
    #     udot_lb = np.array(self.u._mpc_udot_lb)[:,None] # rates of change dont care about current state
    #     udot_ub = np.array(self.u._mpc_udot_ub)[:,None]
            
    #     x_lb = np.tile(x_lb,(hzn,1))    
    #     x_ub = np.tile(x_ub,(hzn,1))
        
    #     state_constr_lower = x_lb - MM @ dx
    #     state_constr_upper = x_ub - MM @ dx
        
    #     # the state constraint input sequence matrix is just CC
        
    #     # calculate the command saturation limits vector
        
    #     cmd_constr_lower = np.tile(u_lb,(hzn,1))
    #     cmd_constr_upper = np.tile(u_ub,(hzn,1))
        
    #     # calculate the command saturation input sequence matrix -> just eye
        
    #     cmd_constr_mat = np.eye(n*hzn)
        
    #     # calculate the command rate saturation limits vector
        
    #     # needs to be relative to the linearisation point (zero)
    #     u0_rate_constr_lower = udot_lb * self.paras.dt
    #     u0_rate_constr_upper = udot_ub * self.paras.dt
        
    #     cmd_rate_constr_lower = np.concatenate((u0_rate_constr_lower,np.tile(udot_lb,(hzn-1,1))))
    #     cmd_rate_constr_upper = np.concatenate((u0_rate_constr_upper,np.tile(udot_ub,(hzn-1,1))))
        
    #     # calculate the command rate saturation input sequence matrix
        
    #     cmd_rate_constr_mat = np.eye(n*hzn)
    #     for i in range(n*hzn):
    #         if i >= n:
    #             cmd_rate_constr_mat[i,i-n] = -1
                
    #     # assemble the complete matrices to send to OSQP
                
    #     OSQP_A = np.concatenate((CC, cmd_constr_mat, cmd_rate_constr_mat), axis=0)
    #     OSQP_l = np.concatenate((state_constr_lower, cmd_constr_lower, cmd_rate_constr_lower))
    #     OSQP_u = np.concatenate((state_constr_upper, cmd_constr_upper, cmd_rate_constr_upper))
        
    #     # test unconstrained performance:
            
    #     if unconstrained:
        
    #         OSQP_A = np.zeros([m*hzn + 2*n*hzn,n*hzn])
    #         OSQP_l = np.ones([m*hzn + 2*n*hzn,1]) * -np.inf
    #         OSQP_u = np.ones([m*hzn + 2*n*hzn,1]) * np.inf
            
            
    #     OSQP_P = csc_matrix(OSQP_P)
    #     OSQP_A = csc_matrix(OSQP_A)
        
    #     # return OSQP_P, OSQP_u, OSQP_A, OSQP_l, OSQP_q
        
    #     m = osqp.OSQP()
    #     m.setup(P=OSQP_P, q=OSQP_q, A=OSQP_A, l=OSQP_l, u=OSQP_u)
        
    #     results = m.solve()
        
    #     return results, H, F, G, CC, MM
    