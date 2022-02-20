import torch
import numpy as np
from scipy.optimize import minimize
from numpy import pi
from dynamics.nlplant import calc_xdot

def trim(h_t, v_t, x, u):

    print('THIS FUNCTION DOES NOT TRIM CORRECTLY AS OF TIME OF WRITING')

    print("I am like 99% sure it is because of the torch clips being used in the objective function")
    print("as opposed to the np.clip as before, the optimal thrust was -2000 ish last i checked")
    print("the minimum should be 1000")
    
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
    
    
    # initial guesses
    thrust = 5000           # thrust, lbs
    elevator = -0.09        # elevator, degrees
    alpha = 8.49            # AOA, degrees
    rudder = -0.01          # rudder angle, degrees
    aileron = 0.01          # aileron, degrees
    
    UX0 = [thrust, elevator, alpha, rudder, aileron]

    #################### convert everything to numpy #################### 

    #x_np = x.values.numpy()
    #u_np = u.values.numpy()

    opt = minimize(obj_func, UX0, args=((h_t, v_t, x, u)), method='Nelder-Mead',tol=1e-10,options={'maxiter':5e+04})
    
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

    #################### back to torch tensors #####################
    x_trim = torch.tensor([0, 0, h_t, 0, alpha_t, 0, v_t, alpha_t, 0, 0, 0, 0, P3_t, dstab_t, da_t, dr_t, dlef, -alpha_t*180/pi])
    
    return x_trim, opt


def obj_func(UX0, h_t, v_t, x, u):

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
    x.values = torch.tensor([npos, epos, h, phi, alpha, psi, V, alpha, beta, p, q, r, P3, dh, da, dr, dlef, -alpha*180/pi])
    
    # thrust limits
    x.values[12] = torch.clip(x.values[12], u.lower_cmd_bound[0], u.upper_cmd_bound[0])
    # elevator limits
    x.values[13] = torch.clip(x.values[13], u.lower_cmd_bound[1], u.upper_cmd_bound[1])
    # aileron limits
    x.values[14] = torch.clip(x.values[14], u.lower_cmd_bound[2], u.upper_cmd_bound[2])
    # rudder limits
    x.values[15] = torch.clip(x.values[15], u.lower_cmd_bound[3], u.upper_cmd_bound[3])
    # alpha limits
    x.values[7] = torch.clip(x.values[7], x.lower_bound[7]*pi/180, x.upper_bound[7]*pi/180)
    
    import pdb
    pdb.set_trace()
        
    u = x.values[12:16]
    xdot,_,_ = calc_xdot(x.values, u)
    xdot = xdot.reshape([18,1])

    phi_w = 10
    theta_w = 10
    psi_w = 10

    weight = torch.tensor([0, 0, 5, phi_w, theta_w, psi_w, 2, 10, 10, 10, 10, 10], dtype=torch.float32).reshape([1,12])
    cost = torch.mm(weight,xdot[0:12]**2)

    weight = np.array([0, 0, 5, phi_w, theta_w, psi_w, 2, 10, 10, 10, 10, 10], dtype=np.float32).transpose()
    cost = np.matmul(weight,(xdot[0:12]**2).numpy())
    return cost.numpy().squeeze()
