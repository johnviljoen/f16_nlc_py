import torch

class Get_observation():
    
    def __init__(self, obs_idx):
        self.obs_idx = obs_idx

    def __call__(self, x, u):
        
        """
        args:
            x:
                {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, lf2, lf1}
            u:
                {T, dh, da, dr}
        
        returns:
            mpc_x:
                a torch tensor of the observable states as dictated by self.obs_idx
        """

        return torch.tensor([x[i] for i in self.obs_idx])
        
