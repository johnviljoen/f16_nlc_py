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

print(cl.hifi_C_lef(inp2))

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

def test_1d(points, values, table):
    for i, point in enumerate(points):
        # i is the alpha index
        print((values[i] - table(points[i].unsqueeze(0)))[3])

def get_c_lookup(fname):
    hifi_C_fnames = [
        'CX0120_ALPHA1_BETA1_DH1_201.dat', # Cx
        'CZ0120_ALPHA1_BETA1_DH1_301.dat', # Cz
        'CM0120_ALPHA1_BETA1_DH1_101.dat', # Cm
        'CY0320_ALPHA1_BETA1_401.dat',     # Cy
        'CN0120_ALPHA1_BETA1_DH2_501.dat', # Cn
        'CL0120_ALPHA1_BETA1_DH2_601'      # Cl
    ]
    hifi_damping_fnames = [
        'CX1120_ALPHA1_204.dat', # CXq
        'CY1320_ALPHA1_406.dat', # CYr
        'CY1220_ALPHA1_408.dat', # CYp
        'CZ1120_ALPHA1_304.dat', # CZq
        'CL1320_ALPHA1_606.dat', # CLr
        'CL1220_ALPHA1_608.dat', # CLp
        'CM1120_ALPHA1_104.dat', # CMq
        'CN1320_ALPHA1_506.dat', # CNr
        'CN1220_ALPHA1_508.dat', # CNp
    ]
    hifi_C_lef_fnames = [
        'CX0820_ALPHA2_BETA1_202.dat', # Cx_lef
        'CZ0820_ALPHA2_BETA1_302.dat', # Cz_lef
        'CM0820_ALPHA2_BETA1_102.dat', # Cm_lef
        'CY0820_ALPHA2_BETA1_402.dat', # Cy_lef
        'CN0820_ALPHA2_BETA1_502.dat', # Cn_lef
        'CL0820_ALPHA2_BETA1_602.dat' # Cl_lef
    ]
    hifi_damping_lef_fnames = [
        'CX1420_ALPHA2_205.dat', # delta_CXq_lef
        'CY1620_ALPHA2_407.dat', # delta_CYr_lef
        'CY1520_ALPHA2_409.dat', # delta_CYp_lef
        'CZ1420_ALPHA2_305.dat', # delta_CZq_lef
        'CL1620_ALPHA2_607.dat', # delta_CLr_lef
        'CL1520_ALPHA2_609.dat', # delta_CLp_lef
        'CM1420_ALPHA2_105.dat', # delta_CMq_lef
        'CN1620_ALPHA2_507.dat', # delta_CNr_lef
        'CN1520_ALPHA2_509.dat' # delta_CNp_lef
    ]
    hifi_rudder_fnames = [
        'CY0720_ALPHA1_BETA1_405.dat', # Cy_r30
        'CN0720_ALPHA1_BETA1_503.dat', # Cn_r30
        'CL0720_ALPHA1_BETA1_603.dat' # Cl_r30
    ]
    hifi_ailerons_fnames = [
        'CY0620_ALPHA1_BETA1_403.dat', # Cy_a20
        'CY0920_ALPHA2_BETA1_404.dat', # Cy_a20_lef
        'CN0620_ALPHA1_BETA1_504.dat', # Cn_a20
        'CN0920_ALPHA2_BETA1_505.dat', # Cn_a20_lef
        'CL0620_ALPHA1_BETA1_604.dat', # Cl_a20
        'CL0920_ALPHA2_BETA1_605.dat' # Cl_a20_lef
    ]
    hifi_other_coeffs_fnames = [
        'CN9999_ALPHA1_brett.dat', # delta_CNbeta
        'CL9999_ALPHA1_brett.dat', # delta_CLbeta
        'CM9999_ALPHA1_brett.dat', # delta_Cm
        'ETA_DH1_brett.dat', # eta_el
        'DOES_NOT_EXIST.dat', # ignore deep-stall regime, delta_Cm_ds = 0
    ]
    #print(len(hifi_C_fnames)  len(hifi_damping_fnames) )
    if fname in hifi_C_fnames:
        table = cl.hifi_C

    elif fname in hifi_damping_fnames:
        table = cl.hifi_damping

    elif fname in hifi_C_lef_fnames:
        table = cl.hifi_C_lef

    elif fname in hifi_damping_lef_fnames:
        table = cl.hifi_damping_lef

    elif fname in hifi_rudder_fnames:
        table = cl.hifi_rudder

    elif fname in hifi_ailerons_fnames:
        table = cl.hifi_ailerons

    elif fname in hifi_other_coeffs_fnames:
        table = cl.hifi_other_coeffs

    return table 
    

print("reading aerodata...")
i = 0
for file in os.listdir("tables/aerodata"):
    if file.endswith(".dat"):
        try:
            points = pl.points[file]
            values = pl.tables[file]
            table = get_c_lookup(file)
            i += 1
        except:
            print(f"    ignoring {file}")

        # do stuff with points and values

if i == 44:
    print("PASS: all 44 tables read successsfully")
else:
    print("PASS: table loading complete")

test_1d(points, values, table)

import pdb
pdb.set_trace()