"""
NOTE: like the lookup tables themselves, this script can ONLY be called
from the directory above due to the C shared libraries path definitions.
"""

print('------------------------------------------------------')
print('-------------Conducting tables unit tests-------------')
print('------------------------------------------------------')

import torch
from tables.c_tables import c_lookup as cl

inp = torch.tensor([1.0,2.0])
print(cl.hifi_C_lef(inp))