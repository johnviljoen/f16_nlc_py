import torch
from dynamics.aircraft import primary_xdot
from dynamics.actuators import upd_lef, upd_thrust, upd_dstab, upd_ail, upd_rud

def calc_xdot(x, u):
    # initialise variables
    actuator_xdot = torch.zeros(6)
    # Thrust Model
    actuator_xdot[0] = upd_thrust(u[0], x[12])
    # Dstab Model
    actuator_xdot[1] = upd_dstab(u[1], x[13])
    # aileron model
    actuator_xdot[2] = upd_ail(u[2], x[14])
    # rudder model
    actuator_xdot[3] = upd_rud(u[3], x[15])
    # leading edge flap model
    actuator_xdot[5], actuator_xdot[4] = upd_lef(x[2], x[6], x[7], x[17], x[16])
    # run nlplant for xdot
    xdot, accelerations, atmospherics = primary_xdot(x)
    # assign actuator xdots
    xdot[12:18] = actuator_xdot
    return xdot, accelerations, atmospherics

