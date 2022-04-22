import torch
# in this case primary_xdot refers to the primary 12 states of a free flying
# object in space. The rest of the states come from actuators, like the engine
# or the ailerons etc
from dynamics.aircraft import primary_xdot
from dynamics.actuators import upd_lef, upd_thrust, upd_dstab, upd_ail, upd_rud

def calc_xdot(x, u):

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

class Calc_xdot_mpc():

    """
    methods:
        update:
            Resets the non MPC states and inputs to be those provided to it
        forward:
            calculates the MPC state time derivates (xdot_mpc) for the MPC
            states and inputs provided to it
    """

    def __init__(self, std_x, std_u, mpc_x_idx, mpc_u_idx):
        """
        args:
            std_x:
                {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, lf2, lf1}
            std_u:
                {t, dh, da, dr}
            mpc_x_idx:
                list of index of elements of x in x_full
            mpc_u_idx:
                list of index of elements of u in u_full
        """
        self.std_x = std_x
        self.std_u = std_u
        self.mpc_x_idx = mpc_x_idx
        self.mpc_u_idx = mpc_u_idx
        
    def __call__(self, mpc_x, mpc_u):
        return self.forward(mpc_x, mpc_u)

    def update_std_x_u(self, std_x, std_u):
        """    
        args:
            std_x:
                {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, lf2, lf1}
            std_u:
                {t, dh, da, dr}
        """
        self.std_x = std_x
        self.std_u = std_u

    def forward(self, mpc_x, mpc_u):
        
        """
        args:
            mpc_x:
                {h,phi,theta,V,alpha,beta,p,q,r,lf1,lf2}
            mpc_u:
                {dh,da,dr}

        returns:
            xdot:
                time derivatives of {h,phi,theta,alpha,beta,p,q,r,lf1,lf2}
        """

        # without assertions we just get a seg fault if wrong states input, this is much easier to debug
        assert len(mpc_x) == 9, f"ERROR: expected 9 states, got {len(x)}"
        assert len(mpc_u) == 3, f"ERROR: expected 3 inputs, got {len(u)}"

        # take the current full state as the starting point, and add the mpc states 
        for mpc_i, std_i in enumerate(self.mpc_x_idx):
            self.std_x[std_i] = mpc_x[mpc_i]
       
        for mpc_i, std_i in enumerate(self.mpc_u_idx):
            self.std_u[std_i] = mpc_u[mpc_i]

        std_xdot = calc_xdot(self.std_x, self.std_u)[0]
        mpc_xdot = torch.zeros(len(mpc_x))
        for mpc_i, std_i in enumerate(self.mpc_x_idx):
            mpc_xdot[mpc_i] = std_xdot[std_i]
        
        return mpc_xdot






