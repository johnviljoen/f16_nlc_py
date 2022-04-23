import torch
from scipy.signal import cont2discrete

class Linmod():

    """
    Function to linearise any general input output system numerically. This should
    generally be considered a last resort as significant parts of systems can usually be
    linearised analytically, which is much faster. However, in this case I just
    need something that I know works.

    after instantiation, the forward function takes the states and the inputs, and
    returns the discrete time A,B,C,D matrices
    """

    def __init__(self, calc_xdot, get_obs, num_states, num_inps, num_obs_states, eps, dt):
        self.calc_xdot = calc_xdot
        self.get_obs = get_obs
        self.eps = eps
        self.dt = dt
        self.num_states = num_states
        self.num_inps = num_inps
        
        self.A = torch.zeros([num_states, num_states])
        self.B = torch.zeros([num_states, num_inps])
        self.C = torch.zeros([num_obs_states, num_states])
        self.D = torch.zeros([num_obs_states, num_inps])

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        # Perturb each of the state variables and compute linearisation
        for i in range(self.num_states):

            dx = torch.zeros(self.num_states)
            dx[i] = self.eps

            self.A[:,i] = (self.calc_xdot(x + dx, u)[0] - self.calc_xdot(x, u)[0]) / self.eps
            self.C[:,i] = (self.get_obs(x + dx, u) - self.get_obs(x, u)) / self.eps
        for i in range(self.num_inps):

            du = torch.zeros(self.num_inps)
            du[i] = self.eps

            self.B[:,i] = (self.calc_xdot(x, u + du)[0] - self.calc_xdot(x, u)[0]) / self.eps
            self.D[:,i] = (self.get_obs(x, u + du) - self.get_obs(x, u)) / self.eps


        return cont2discrete((self.A, self.B, self.C, self.D), self.dt)[0:4]
