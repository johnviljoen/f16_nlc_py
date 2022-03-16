"""
NOTE: like the lookup tables themselves, this script can ONLY be called
from the directory above due to the C shared libraries path definitions.

Validation Plan:
    1. Check that all exact points in the tables are correct
    2. Check that all limits are correct (no extrapolation)
    3. Sample array randomly and ensure same to a tolerance of 1e-07
"""

print('------------------------------------------------------')
print('-------------Conducting tables unit tests-------------')
print('------------------------------------------------------')

import torch
import os
from tables.c_tables import c_lookup as cl
from tables.py_tables import py_lookup as pl

inp1 = torch.tensor([1.0])
inp2 = torch.tensor([1.0,2.0])
inp3 = torch.tensor([1.0,2.0,3.0])

#print(cl.hifi_C_lef(inp2))

"""
1.
    The files will with one exception have lookup axes of (alpha, beta, dh) in that order.
    The exception to the rule only has dh. These points are contained in py_lookup.points 
    dictionary.

    I will validate the c_lookup at the exact points of the table first by going through
    the aerodata folder manually here in Python.
"""

def test_3d(points, values):
    pass

def test_1d(points, values):
    for i, point in enumerate(points):
        # i is the alpha index
        values[i] # - cl

def get_c_lookup(coeff):
    if coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
        table = cl.hifi_C
    elif coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
        table = cl.hifi_damping
    elif coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
        table = cl.hifi_C_lef
    elif coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
        table = cl.hifi_damping_lef
    elif coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
        table = cl.hifi_rudder
    elif coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
        table = cl.hifi_ailerons
    elif coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
        table = cl.hifi_other_coeffs
    return table

def get_coeff(fname):
    hifi_C_fnames = [
        'CX0120_ALPHA1_BETA1_DH1_201.dat',
        'CZ0120_ALPHA1_BETA1_DH1_301.dat',
        'CM0120_ALPHA1_BETA1_DH1_101.dat',
        'CY0320_ALPHA1_BETA1_401.dat',
        'CN0120_ALPHA1_BETA1_DH2_501.dat',
        'CL0120_ALPHA1_BETA1_DH2_601'
    ]
    hifi_damping_fnames = [
    
    ]
    hifi_C_lef_fnames = [
        'CX0820_ALPHA2_BETA1_202.dat',
        'CZ0820_ALPHA2_BETA1_302.dat',
        'CM0820_ALPHA2_BETA1_102.dat',
        'CY0820_ALPHA2_BETA1_402.dat',
        'CN0820_ALPHA2_BETA1_502.dat',
        'CL0820_ALPHA2_BETA1_602.dat',
    ]
    hifi_damping_lef_fnames = []
    hifi_rudder_fnames = []
    hifi_ailerons_fnames = []
    hifi_other_coeffs_fnames = []

    if fname in hifi_C_fnames:
        table = cl.hifi_C
    elif fname in hifi_damping_fnames:
        pass

print("reading aerodata...")
i = 0
for file in os.listdir("tables/aerodata"):
    if file.endswith(".dat"):
        try:
            points = pl.points[file]
            values = pl.tables[file]
            i += 1
        except:
            print(f"    ignoring {file}")

        # do stuff with points and values

if i == 44:
    print("PASS: all 44 tables read successsfully")

import pdb
pdb.set_trace()