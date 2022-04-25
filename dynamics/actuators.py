import ctypes
import torch
from numpy import pi
from dynamics.parameters import x_lb, x_ub, u_lb, u_ub
from dynamics.aircraft import atmos

def upd_lef(h, V, alpha, lef_state_1, lef_state_2):
    
    coeff = atmos(h, V)
    atmos_out = coeff[1]/coeff[2] * 9.05
    alpha_deg = alpha*180/pi
    
    LF_err = alpha_deg - (lef_state_1 + (2 * alpha_deg))
    #lef_state_1 += LF_err*7.25*time_step
    LF_out = (lef_state_1 + (2 * alpha_deg)) * 1.38
    
    lef_cmd = LF_out + 1.45 - atmos_out
    
    # command saturation
    lef_cmd = torch.clip(lef_cmd,x_lb[16],x_ub[16])
    # rate saturation
    lef_err = torch.clip((1/0.136) * (lef_cmd - lef_state_2),-25,25)
    
    return LF_err*7.25, lef_err

def upd_thrust(T_cmd, T_state):
    # command saturation
    T_cmd = torch.clip(T_cmd,u_lb[0],u_ub[0])
    # rate saturation
    return torch.clip(T_cmd - T_state, -10000, 10000)

def upd_dstab(dstab_cmd, dstab_state):
    # command saturation
    dstab_cmd = torch.clip(dstab_cmd,u_lb[1],u_ub[1])
    # rate saturation
    return torch.clip(20.2*(dstab_cmd - dstab_state), -60, 60)

def upd_ail(ail_cmd, ail_state):
    # command saturation
    ail_cmd = torch.clip(ail_cmd,u_lb[2],u_ub[2])
    # rate saturation
    return torch.clip(20.2*(ail_cmd - ail_state), -80, 80)

def upd_rud(rud_cmd, rud_state):
    # command saturation
    rud_cmd = torch.clip(rud_cmd,u_lb[3],u_ub[3])
    # rate saturation
    return torch.clip(20.2*(rud_cmd - rud_state), -120, 120)

