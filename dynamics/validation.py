print('------------------------------------------------------')
print('------------Conducting dynamics unit tests------------')
print('------------------------------------------------------')

import torch

try:
    from dynamics.parameters import state_vector
except:
    print('FAIL: failed to import state vector from parameters')

# print('PASS: imported state vector from parameters')

xu = state_vector.values

try:
    from dynamics.aircraft import calc_xdot
except:
    print('FAIL: failed to import calc_xdot from nlplant')
    print('maybe you changed its name?')

# print('PASS: import calc_xdot from nlplant')

try:
    xdot = calc_xdot(torch.tensor(xu))
except:
    print('FAIL: failed to generate an xdot from calc_xdot using the imported state vector')

