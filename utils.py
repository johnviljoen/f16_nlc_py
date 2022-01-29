#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:45:08 2021

@author: johnviljoen
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import scipy
import ctypes
import os
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d


from parameters import u_lb, u_ub, x_lb, x_ub
# from control.matlab import *


# In[]


def construct_lookup(aerodata_path):
    
    # retrieve aerodata
    aerodata, ndinfo = parse_aerodata('aerodata')

    # instantiate ranges
    alpha1 = aerodata['ALPHA1']
    alpha2 = aerodata['ALPHA2']
    beta1 = aerodata['BETA1']
    dele1 = aerodata['DH1']
    dele2 = aerodata['DH2']

    # instantiate lookup dictionary
    lookup = {}

    # run through all aerodata files and assign them to the correct lookup key
    # there appear to be 43 used of the 49 I believe
    for file in aerodata.keys():

        # 3D tables in alpha, beta, and dele
        if file == 'CX0120_ALPHA1_BETA1_DH1_201':
            lookup['Cx'] = rgi((alpha1,beta1,dele1),aerodata[file])
        elif file == 'CZ0120_ALPHA1_BETA1_DH1_301':
            lookup['Cz'] = rgi((alpha1,beta1,dele1),aerodata[file])    
        elif file == 'CM0120_ALPHA1_BETA1_DH1_101':
            lookup['Cm'] = rgi((alpha1,beta1,dele1),aerodata[file])
        elif file == 'CY0320_ALPHA1_BETA1_401':
            lookup['Cy'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0120_ALPHA1_BETA1_DH2_501':
            lookup['Cn'] = rgi((alpha1,beta1,dele2),aerodata[file])
        elif file == 'CL0120_ALPHA1_BETA1_DH2_601':
            lookup['Cl'] = rgi((alpha1,beta1,dele2),aerodata[file])


        elif file == 'CX0820_ALPHA2_BETA1_202':
            lookup['Cx_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CZ0820_ALPHA2_BETA1_302':
            lookup['Cz_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CM0820_ALPHA2_BETA1_102':
            lookup['Cm_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CY0820_ALPHA2_BETA1_402':
            lookup['Cy_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CN0820_ALPHA2_BETA1_502':
            lookup['Cn_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CL0820_ALPHA2_BETA1_602':
            lookup['Cl_lef'] = rgi((alpha2,beta1),aerodata[file])
        
        elif file == 'CX1120_ALPHA1_204':
            lookup['CXq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CZ1120_ALPHA1_304':
            lookup['CZq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CM1120_ALPHA1_104':
            lookup['CMq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CY1220_ALPHA1_408':
            lookup['CYp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CY1320_ALPHA1_406':
            lookup['CYr'] = interp1d(alpha1,aerodata[file])
        elif file == 'CN1320_ALPHA1_506':
            lookup['CNr'] = interp1d(alpha1,aerodata[file])
        elif file == 'CN1220_ALPHA1_508':
            lookup['CNp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL1220_ALPHA1_608':
            lookup['CLp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL1320_ALPHA1_606':
            lookup['CLr'] = interp1d(alpha1,aerodata[file])

        elif file == 'CX1420_ALPHA2_205':
            lookup['delta_CXq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CY1620_ALPHA2_407':
            lookup['delta_CYr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CY1520_ALPHA2_409':
            lookup['delta_CYp_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CZ1420_ALPHA2_305':
            lookup['delta_CZq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CL1620_ALPHA2_607':
            lookup['delta_CLr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CL1520_ALPHA2_609':
            lookup['delta_CLp_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CM1420_ALPHA2_105':
            lookup['delta_CMq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CN1620_ALPHA2_507':
            lookup['delta_CNr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CN1520_ALPHA2_509':
            lookup['delta_CNp_lef'] = interp1d(alpha2,aerodata[file])

        elif file == 'CY0720_ALPHA1_BETA1_405':
            lookup['Cy_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0720_ALPHA1_BETA1_503':
            lookup['Cn_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CL0720_ALPHA1_BETA1_603':
            lookup['Cl_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CY0620_ALPHA1_BETA1_403':
            lookup['Cy_a20'] = rgi((alpha1,beta1),aerodata[file])

        elif file == 'CY0920_ALPHA2_BETA1_404':
            lookup['Cy_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        
        elif file == 'CN0620_ALPHA1_BETA1_504':
            lookup['Cn_a20'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0920_ALPHA2_BETA1_505':
            lookup['Cn_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CL0620_ALPHA1_BETA1_604':
            lookup['Cl_a20'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CL0920_ALPHA2_BETA1_605':
            lookup['Cl_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CN9999_ALPHA1_brett':
            lookup['delta_CNbeta'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL9999_ALPHA1_brett':
            lookup['delta_CLbeta'] = interp1d(alpha1,aerodata[file])
        elif file == 'CM9999_ALPHA1_brett':
            lookup['delta_Cm'] = interp1d(alpha1,aerodata[file])
        elif file == 'ETA_DH1_brett':
            lookup['eta_el'] = interp1d(dele1,aerodata[file])
    
    # hifi_C_lef
    lookup['delta_Cx_lef'] = lambda alpha, beta: lookup['Cx_lef']((alpha,beta)) - lookup['Cx']((alpha,beta,0))
    lookup['delta_Cz_lef'] = lambda alpha, beta: lookup['Cz_lef']((alpha,beta)) - lookup['Cz']((alpha,beta,0))
    lookup['delta_Cm_lef'] = lambda alpha, beta: lookup['Cm_lef']((alpha,beta)) - lookup['Cm']((alpha,beta,0))
    lookup['delta_Cy_lef'] = lambda alpha, beta: lookup['Cy_lef']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cn_lef'] = lambda alpha, beta: lookup['Cn_lef']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cl_lef'] = lambda alpha, beta: lookup['Cl_lef']((alpha,beta)) - lookup['Cl']((alpha,beta,0))

    # hifi_rudder
    lookup['delta_Cy_r30'] = lambda alpha, beta: lookup['Cy_r30']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cn_r30'] = lambda alpha, beta: lookup['Cn_r30']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cl_r30'] = lambda alpha, beta: lookup['Cl_r30']((alpha,beta)) - lookup['Cl']((alpha,beta,0))

    # hifi_ailerons
    lookup['delta_Cy_a20'] = lambda alpha, beta: lookup['Cy_a20']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cy_a20_lef'] = lambda alpha, beta: lookup['Cy_a20_lef']((alpha,beta)) - lookup['Cy_lef']((alpha,beta)) - lookup['delta_Cy_a20'](alpha,beta)
    lookup['delta_Cn_a20'] = lambda alpha, beta: lookup['Cn_a20']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cn_a20_lef'] = lambda alpha, beta: lookup['Cn_a20_lef']((alpha,beta)) - lookup['Cn_lef']((alpha,beta)) - lookup['delta_Cn_a20'](alpha,beta)
    lookup['delta_Cl_a20'] = lambda alpha, beta: lookup['Cl_a20']((alpha,beta)) - lookup['Cl']((alpha,beta,0))
    lookup['delta_Cl_a20_lef'] = lambda alpha, beta: lookup['Cl_a20_lef']((alpha,beta)) - lookup['Cl_lef']((alpha,beta)) - lookup['delta_Cl_a20'](alpha,beta)

    return lookup



def parse_aerodata(fp):

    '''
    As you might expect from its name this function parses the aerodynamic data
    found in the aerodata file. What you might not expect is that it will produce
    a nice pythonic dictionary for accessing these files.
    '''

    # isolate the aerodata directory and list the files
    aerodata_path = fp
    files = os.listdir(aerodata_path)

    # read in the raw .dat files
    aerodata_raw = {}
    for file in files:
        aerodata_raw[os.path.splitext(file)[0]] = open(aerodata_path + '/' + file)

    '''
    There are two sets of aerodynamic datapoints - broadly they can be thought
    of as high fidelity and low fidelity. There exists two sets of alpha and
    dele values. Sometimes they are also mixed.

    For example we could have a set of high fidelity alpha points with low
    fidelity dele points.

    Below I manually entered the number of possible points of alpha and beta
    and dele, and then created a dictionary 'ndinfo' containing the dimensionality
    information of every possible combination of them.
    '''
    ndinfo = {}
    ndinfo['alpha1_points'] = 20
    ndinfo['alpha2_points'] = 14
    ndinfo['beta1_points'] = 19
    ndinfo['dele1_points'] = 5
    ndinfo['dele2_points'] = 3

    # 3D tables
    # here a1b1d1 refers to using the alpha1 beta1 and dele1 point sets
    ndinfo['3D_a1b1d1_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points'], ndinfo['dele1_points']]
    ndinfo['3D_a1b1d1_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points'] * ndinfo['dele1_points']
    ndinfo['3D_a1b1d2_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points'], ndinfo['dele2_points']]
    ndinfo['3D_a1b1d2_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points'] * ndinfo['dele2_points']

    # 2D tables
    # similarly a1b1 refers to alpha1 and beta1 point sets
    ndinfo['2D_a1b1_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points']]
    ndinfo['2D_a1b1_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points']
    ndinfo['2D_a2b1_shape'] = [ndinfo['alpha2_points'], ndinfo['beta1_points']]
    ndinfo['2D_a2b1_size'] = ndinfo['alpha2_points'] * ndinfo['beta1_points']
    ndinfo['2D_a1d1_shape'] = [ndinfo['alpha1_points'],  ndinfo['dele1_points']]
    ndinfo['2D_a1d1_size'] = ndinfo['alpha1_points'] * ndinfo['dele1_points']

    # 1D tables
    ndinfo['1D_a1_shape'] = ndinfo['alpha1_points']
    ndinfo['1D_a1_size'] = ndinfo['alpha1_points']
    ndinfo['1D_d1_shape'] = ndinfo['dele1_points']
    ndinfo['1D_d1_size'] = ndinfo['dele1_points']

    # instantiate dictionaries
    aerodata_list = {}
    aerodata = {}

    # reshape data into aerodata dictionary
    for i in aerodata_raw:
        # get all the dat files in 1D lists
        aerodata_list[i] = [float(x) for x in aerodata_raw[i].read().split()]
        # we need to reshape these 1D lists into 1D, 2D, or 3D tables
        # if its an alpha1, beta1 and dele1 table
        if len(aerodata_list[i]) == ndinfo['3D_a1b1d1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['3D_a1b1d1_shape'])
        # if its an alpha1, beta1 and dele2 table
        elif len(aerodata_list[i]) == ndinfo['3D_a1b1d2_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['3D_a1b1d2_shape'])
        # if its an alpha1 and beta1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a1b1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a1b1_shape'])
        # if its an alpha2 and beta1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a2b1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a2b1_shape'])

        # if its an alpha1 and dele1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a1d1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a1d1_shape'])
        # if its a 1D table
        else:
            aerodata[i] = np.array(aerodata_list[i])

    return aerodata, ndinfo



def setup_OSQP(x_ref, A, B, Q, R, hzn, dt, x, act_states, x_lb, x_ub, u_lb, u_ub, udot_lb, udot_ub):
    
    """
    Function that builds a model predictive control problem from a discrete linear
    state space system (A,B), its corresponding cost function weights (Q,R), and
    its constraints. This is a tracking controller (x_ref) that aims for zero error
    in p,q,r angular rates.
    
    The output is designed for the OSQP constrained quadratic solver (https://osqp.org/docs/)
    
    The function is designed to eventually be generalisable with the rest of the 
    code, but this is not yet complete. This is the reason for some of the unusual
    decisions at first glance, like some input vectors being horizontal 1D and others
    vertical 2D.
    
    args:
        xref:
            1D numpy array -> horizontal vector (0 x m*hzn)
        A:
            2D numpy array (m x m)
        B:
            2D numpy array (m x n)
        Q:
            2D numpy array (m x m)
        R:
            2D numpy array (n x n)
        hzn:
            int
        dt:
            float
        x:
            1D numpy array -> horizontal vector (0 x m)
        act_states:
            1D numpy array -> horizontal vector (0 x n)
        x_lb:
            2D numpy array -> vertical vector (m x 1)
        x_ub:
            2D numpy array -> vertical vector (m x 1)
        u_lb:
            2D numpy array -> vertical vector (n x 1)
        u_ub:
            2D numpy array -> vertical vector (n x 1)      
        udot_lb:
            2D numpy array -> vertical vector (n x 1)
        udot_ub:
            2D numpy array -> vertical vector (n x 1)
            
    returns:
        OSQP_P:
            2D numpy array (n*hzn x n*hzn)
        OSQP_q:
            2D numpy array -> vertical vector (n*hzn x 1)
        OSQP_A:
            2D numpy array (m*hzn x n*hzn)
        OSQP_l:
            2D numpy array -> vertical vector (n*hzn x 1)
        OSQP_u:
            2D numpy array -> vertical vector (n*hzn x 1)
            
    """
    
    m = len(x)                      # number of states
    n = len(act_states)             # number of inputs
    
    x = x[:,None]                   # convert x to vertical vector
    
    x_ref = np.tile(x_ref, hzn)     # stack x_refs to create sequence
    x_ref = x_ref[:,None]     # convert stacked x_ref to vertical vector
    
    # calculate matrices for predictions (p16 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    MM, CC = calc_MC(A, B, hzn)
    
    # calculate LQR gain matrix for mode 2 (https://github.com/python-control/python-control/issues/359)
    
    K = - dlqr(A, B, Q, R)
    
    # calculate terminal weighting matrix (p24 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    Q_bar = scipy.linalg.solve_discrete_lyapunov((A + B @ K).T, Q + K.T @ R @ K)
        
    # construct full QQ, RR (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    QQ = dmom(Q, hzn)
    QQ[-m:,-m:] = Q_bar
    RR = dmom(R, hzn)
    
    # construct objective function (2.3) (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    # and implement this in OSQP format
    
    OSQP_P = 2 * (CC.T @ QQ @ CC + RR)
    OSQP_q = -2 * ((x_ref - MM @ x).T @ QQ @ CC).T
    
    """
    There are three constraints to be enforced on the system:
        
        state constraints:
            x(n+1) = Ax(n) + Bu(n)
            
        input command limit constraints:
            u_min <= u <= u_max
            
        input command rate limit constraints:
            udot_min <= udot <= udot_max
    """
    
    # calculate state constraint limits vector
        
    x_lb = np.tile(x_lb,(hzn,1))    
    x_ub = np.tile(x_ub,(hzn,1))
    
    state_constr_lower = x_lb - MM @ x
    state_constr_upper = x_ub - MM @ x
    
    # the state constraint input sequence matrix is just CC
    
    # calculate the command saturation limits vector
    
    cmd_constr_lower = np.tile(u_lb,(hzn,1))
    cmd_constr_upper = np.tile(u_ub,(hzn,1))
    
    # calculate the command saturation input sequence matrix -> just eye
    
    cmd_constr_mat = np.eye(n*hzn)
    
    # calculate the command rate saturation limits vector
    
    u0_rate_constr_lower = act_states[:,None] + udot_lb * dt
    u0_rate_constr_upper = act_states[:,None] + udot_ub * dt
    
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
    
    return OSQP_P, OSQP_q, OSQP_A, OSQP_l, OSQP_u    

# In[ Calculate the MM, CC matrices -> squiggly M and squiggly C in the notes ]

def calc_MC(A, B, hzn, includeFirstRow=False):
    
    # hzn is the horizon
    nstates = A.shape[0]
    ninputs = B.shape[1]
    
    # x0 is the initial state vector of shape (nstates, 1)
    # u is the matrix of input vectors over the course of the prediction of shape (ninputs,horizon)
    
    # initialise CC, MM, Bz
    CC = np.zeros([nstates*hzn, ninputs*hzn])
    MM = np.zeros([nstates*hzn, nstates])
    Bz = np.zeros([nstates, ninputs])
        
    for i in range(hzn):
        MM[nstates*i:nstates*(i+1),:] = np.linalg.matrix_power(A,i+1) 
        for j in range(hzn):
            if i-j >= 0:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = np.matmul(np.linalg.matrix_power(A,(i-j)),B)
            else:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = Bz
                
    if includeFirstRow:
        CC = np.concatenate((np.zeros([nstates,ninputs*hzn]), CC), axis=0)
        MM = np.concatenate((np.eye(nstates), MM), axis=0)

    return MM, CC

def calc_HFG(A, B, QQ, RR):
    
    hzn = np.rint((RR.shape[1]/B.shape[1])).astype(np.int64)
    
    
    MM, CC = calc_MC(A,B,hzn)
        
    H = CC.T @ QQ @ CC + RR
    F = CC.T @ QQ @ MM
    G = MM.T @ QQ @ MM
    
    return H, F, G

def calc_QR(n, C, hzn, Qbar_eq_Q=True):
    
    Q = C.T @ C
    R = np.eye(n) * 0.01
    
    QQ = dmom(Q, hzn)
    
    if C.ndim == 1:
        
        m = 1
        
    else:
    
        m = C.shape[1]
    
    if Qbar_eq_Q:
        
        QQ[-m:,-m:] = Q
    RR = dmom(R, hzn)
    
    return QQ, RR

def calc_L(H,F,n,hzn):
    
    pre_mult = np.zeros([n,n*hzn])
    pre_mult[0:n,0:n] = np.eye(n)
    
    return - pre_mult @ np.linalg.inv(H) @ F




# In[]

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def discrete_gram(A,B,C):
    # source: https://uk.mathworks.com/help/control/ref/ss.gram.html;jsessionid=e2d01bb6a726a08d45c48257df5c
    # controllability gramiam, an nxn matrix, Wc ~ C @ C.T for discrete system (empirical finding not just here)
    Wc = scipy.linalg.solve_discrete_lyapunov(A,B@B.T)
    # observation gramiam, an mxm matrix
    Wo = scipy.linalg.solve_discrete_lyapunov(A,C.T@C)
    # recall that a system is stabilizable if and only if all unstable (and lightly damped) eigenvectors of A are in my controllable subspace
    # meaning they are in the column space of the controllability matrix. So I want to choose my actuators, B s.t. the unstable dynamics directions correspond to big singular vectors in the controllability matrix 
    return Wc, Wo

def ctrb(A,B):
    
    m = A.shape[0] # number of states
    n = B.shape[1] # number of inputs
    
    ctrb_mat = np.zeros((m,n*m))
    
    # [B AB A^2B ... A^(m-1)B]
    
    for i in range(m):
        
        ctrb_mat[:,n*i:n*(i+1)] = np.linalg.matrix_power(A,i) @ B
        
    return ctrb_mat

def is_ctrb(A,B):
    if np.linalg.matrix_rank(ctrb(A,B)) == A.shape[0]:
        return True
    elif np.linalg.matrix_rank(ctrb(A,B)) <= A.shape[0]:
        return False
    
def is_obsv(A,C):
    if np.linalg.matrix_rank(obsv(A,C)) == A.shape[0]:
        return True
    elif np.linalg.matrix_rank(obsv(A,C)) <= A.shape[0]:
        return False

# In[ discrete linear quadratic regulator ]
# from https://github.com/python-control/python-control/issues/359:
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    
    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*(x_ref[k] - x[k])
    select the states that you want considered and make x[k] the difference
    between the current x and the desired x.
      
    How to apply the function:    
        K = dlqr(A_d,B_d,Q,R)
      
    Inputs:
      A_d, B_d, Q, R  -> all numpy arrays  (simple float number not allowed)
      
    Returns:
      K: state feedback gain
    
    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ P @ B+R) @ (B.T @ P @ A))
    return K

# In[ degenerate 2D square matrix ]

def square_mat_degen_2d(mat, degen_idx):
    
    degen_mat = np.zeros([len(degen_idx),len(degen_idx)])
    
    for i in range(len(degen_idx)):
        
        degen_mat[:,i] = mat[degen_idx, [degen_idx[i] for x in range(len(degen_idx))]]
        
    return degen_mat

# In[ diagonal matrix of matrices ]

# def gmim(mat, submat_dims):
#     # get matrix in matrix -> gmim
    
#     mat[]

# def dmom(mat, num_mats):
#     return scipy.linalg.block_diag(np.kron(np.eye(num_mats), mat))

# a much faster implementation: (~40x faster)
def dmom(mat, num_mats):
    # diagonal matrix of matrices -> dmom
    
    if isinstance(mat, np.ndarray):
    
        # dimension extraction
        nrows = mat.shape[0]
        ncols = mat.shape[1]
        
        # matrix of matrices matomats -> I thought it sounded cool
        matomats = np.zeros((nrows*num_mats,ncols*num_mats))
        
        for i in range(num_mats):
            for j in range(num_mats):
                if i == j:
                    matomats[nrows*i:nrows*(i+1),ncols*j:ncols*(j+1)] = mat
                    
    else:
        
        matomats = np.eye(num_mats) * mat
                
    return matomats

# In[ calculate actuator model time derivatives ]

def upd_lef(h, V, coeff, alpha, lef_state_1, lef_state_2, nlplant):
    
    nlplant.atmos(ctypes.c_double(h),ctypes.c_double(V),ctypes.c_void_p(coeff.ctypes.data))
    atmos_out = coeff[1]/coeff[2] * 9.05
    alpha_deg = alpha*180/pi
    
    LF_err = alpha_deg - (lef_state_1 + (2 * alpha_deg))
    #lef_state_1 += LF_err*7.25*time_step
    LF_out = (lef_state_1 + (2 * alpha_deg)) * 1.38
    
    lef_cmd = LF_out + 1.45 - atmos_out
    
    # command saturation
    lef_cmd = np.clip(lef_cmd,x_lb[16],x_ub[16])
    # rate saturation
    lef_err = np.clip((1/0.136) * (lef_cmd - lef_state_2),-25,25)
    
    return LF_err*7.25, lef_err

def upd_thrust(T_cmd, T_state):
    # command saturation
    T_cmd = np.clip(T_cmd,u_lb[0],u_ub[0])
    # rate saturation
    return np.clip(T_cmd - T_state, -10000, 10000)

def upd_dstab(dstab_cmd, dstab_state):
    # command saturation
    dstab_cmd = np.clip(dstab_cmd,u_lb[1],u_ub[1])
    # rate saturation
    return np.clip(20.2*(dstab_cmd - dstab_state), -60, 60)

def upd_ail(ail_cmd, ail_state):
    # command saturation
    ail_cmd = np.clip(ail_cmd,u_lb[2],u_ub[2])
    # rate saturation
    return np.clip(20.2*(ail_cmd - ail_state), -80, 80)

def upd_rud(rud_cmd, rud_state):
    # command saturation
    rud_cmd = np.clip(rud_cmd,u_lb[3],u_ub[3])
    # rate saturation
    return np.clip(20.2*(rud_cmd - rud_state), -120, 120)

# In[ replicate MATLAB tic toc functions ]

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    

# In[]

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)
    
# In[ visualise full 18 DoF system time history ]

def vis_mpc_u(u_storage, rng):
    fig, axs = plt.subplots(3,1)
    
    # axs[0].plot(rng, u_storage[:,0])
    # axs[0].set_ylabel('T_cmd')
    
    axs[0].plot(rng, u_storage[:,0])
    axs[0].set_ylabel('dh_cmd')
    
    axs[1].plot(rng, u_storage[:,1])
    axs[1].set_ylabel('da_cmd')
    
    axs[2].plot(rng, u_storage[:,2])
    axs[2].set_ylabel('dr_cmd')
    
def vis_mpc_x(x_storage, rng):
    
    fig1, axs1 = plt.subplots(2, 1)
    #fig.suptitle('Vertically stacked subplots')
        
    axs1[0].plot(rng, x_storage[:,0])
    axs1[0].set_ylabel('phi (rad)')
    
    axs1[1].plot(rng, x_storage[:,1])
    axs1[1].set_ylabel('theta (rad)')
    
    fig2, axs2 = plt.subplots(2,1)
    
    axs2[0].plot(rng, x_storage[:,2]*180/pi)
    axs2[0].set_ylabel('alpha (deg)')
    
    axs2[1].plot(rng, x_storage[:,3]*180/pi)
    axs2[1].set_ylabel('beta (deg)')
    
    fig3, axs3 = plt.subplots(3,1)
    
    axs3[0].plot(rng, x_storage[:,4]*180/pi)
    axs3[0].set_ylabel('p (deg/s)')
    
    axs3[1].plot(rng, x_storage[:,5]*180/pi)
    axs3[1].set_ylabel('q (deg/s)')
    
    axs3[2].plot(rng, x_storage[:,6]*180/pi)
    axs3[2].set_ylabel('r (deg/s)')
    axs3[2].set_xlabel('time (s)')

def vis_u(u_storage, rng):
    
    fig, axs = plt.subplots(3,1)
    
    # axs[0].plot(rng, u_storage[:,0])
    # axs[0].set_ylabel('T_cmd')
    
    axs[0].plot(rng, u_storage[:,0])
    axs[0].set_ylabel('dh_cmd')
    
    axs[1].plot(rng, u_storage[:,1])
    axs[1].set_ylabel('da_cmd')
    
    axs[2].plot(rng, u_storage[:,2])
    axs[2].set_ylabel('dr_cmd')
    
def vis_x(x_storage, rng):

    fig, axs = plt.subplots(12, 1)
    #fig.suptitle('Vertically stacked subplots')
    axs[0].plot(rng, x_storage[:,0])
    axs[0].set_ylabel('npos (ft)')
    
    axs[1].plot(rng, x_storage[:,1])
    axs[1].set_ylabel('epos (ft)')
    
    axs[2].plot(rng, x_storage[:,2])
    axs[2].set_ylabel('h (ft)')
    
    axs[3].plot(rng, x_storage[:,3])
    axs[3].set_ylabel('$\phi$ (rad)')
    
    axs[4].plot(rng, x_storage[:,4])
    axs[4].set_ylabel('$\theta$ (rad)')
    
    axs[5].plot(rng, x_storage[:,5])
    axs[5].set_ylabel('$\psi$ (rad)')
    
    axs[6].plot(rng, x_storage[:,6])
    axs[6].set_ylabel("V_t (ft/s)")
    
    axs[7].plot(rng, x_storage[:,7]*180/pi)
    axs[7].set_ylabel('alpha (deg)')
    
    axs[8].plot(rng, x_storage[:,8]*180/pi)
    axs[8].set_ylabel('beta (deg)')
    
    axs[9].plot(rng, x_storage[:,9]*180/pi)
    axs[9].set_ylabel('p (deg/s)')
    
    axs[10].plot(rng, x_storage[:,10]*180/pi)
    axs[10].set_ylabel('q (deg/s)')
    
    axs[11].plot(rng, x_storage[:,11]*180/pi)
    axs[11].set_ylabel('r (deg/s)')
    axs[11].set_xlabel('time (s)')
    
    fig2, axs2 = plt.subplots(5,1)
    
    axs2[0].plot(rng, x_storage[:,12])
    axs2[0].set_ylabel('P3')
    
    axs2[1].plot(rng, x_storage[:,13])
    axs2[1].set_ylabel('dh')
    
    axs2[2].plot(rng, x_storage[:,14])
    axs2[2].set_ylabel('da')
    
    axs2[3].plot(rng, x_storage[:,15])
    axs2[3].set_ylabel('dr')
    
    axs2[4].plot(rng, x_storage[:,16])
    axs2[4].set_ylabel('lef')
