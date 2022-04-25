import torch
import torch.nn as nn

class Atmos(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def __call__(self, alt, vt):
        rho0 = 2.377e-3
        
        tfac =1 - .703e-5*(alt)
        temp = 519.0*tfac
        if alt >= 35000.0: temp=390

        rho=rho0*torch.pow(tfac,4.14)
        mach = vt/torch.sqrt(1.4*1716.3*temp)
        qbar = .5*rho*torch.pow(vt,2)
        ps   = 1715.0*rho*temp

        if ps == 0: ps = 1715
        
        return mach, qbar, ps
