print('------------------------------------------------------')
print('------------Conducting dynamics unit tests------------')
print('------------------------------------------------------')

import torch

try:
    from dynamics.parameters import state_vector
    xu = state_vector.values
except:
    print('FAIL: failed to import state vector from parameters')

# print('PASS: imported state vector from parameters')


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

