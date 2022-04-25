import torch
import torch.nn as nn
from dynamics.actuators import upd_lef, upd_thrust, upd_dstab, upd_ail, upd_rud
from dynamics.eom import EoM

class Nlplant(nn.Module):
    def __init__(self, device, dtype, lookup_type):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.upd_lef = upd_lef
        self.upd_thrust = upd_thrust
        self.upd_dstab = upd_dstab
        self.upd_ail = upd_ail
        self.upd_rud = upd_rud

        self.eom = EoM(device, dtype, lookup_type)

        # for MPC:
        self.std_x = std_x
        self.std_u = std_u
        self.mpc_x_idx = mpc_x_idx
        self.mpc_u_idx = mpc_u_idx

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        """
        args:
            x:
                {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, lf2, lf1}
            u:
                {T, dh, da, dr}

        returns:
            xdot:
                time derivates of x, in same order
            accelerations:
                {anx_cg, any_cg, anz_cg}
            atmospherics:
                {mach, qbar, ps}
        """
        # initialise variables
        actuator_xdot = torch.zeros(6, device=self.device, dtype=self.dtype)
        # Thrust Model
        actuator_xdot[0] = self.upd_thrust(u[0], x[12])
        # Dstab Model
        actuator_xdot[1] = self.upd_dstab(u[1], x[13])
        # aileron model
        actuator_xdot[2] = self.upd_ail(u[2], x[14])
        # rudder model
        actuator_xdot[3] = self.upd_rud(u[3], x[15])
        # leading edge flap model
        actuator_xdot[5], actuator_xdot[4] = self.upd_lef(x[2], x[6], x[7], x[17], x[16])
        # run nlplant for xdot
        xdot, accelerations, atmospherics = self.eom(x)
        # assign actuator xdots
        xdot[12:18] = actuator_xdot
        return xdot, accelerations, atmospherics

    #def mpc(self, x, u):


