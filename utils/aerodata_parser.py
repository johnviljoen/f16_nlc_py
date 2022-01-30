import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d

def construct_lookup(aerodata_path):
    
    # retrieve aerodata
    aerodata, ndinfo = parse_aerodata(aerodata_path)

    # instantiate ranges
    alpha1 = aerodata['ALPHA1']
    alpha2 = aerodata['ALPHA2']
    beta1 = aerodata['BETA1']
    dele1 = aerodata['DH1']
    dele2 = aerodata['DH2']

    # instantiate lookup dictionary
    lookup = {}

    # run through all aerodata files and assign them to the correct lookup key
    # there appear to be 43 used of the 49 I believe
    for file in aerodata.keys():

        # 3D tables in alpha, beta, and dele
        if file == 'CX0120_ALPHA1_BETA1_DH1_201':
            lookup['Cx'] = rgi((alpha1,beta1,dele1),aerodata[file])
        elif file == 'CZ0120_ALPHA1_BETA1_DH1_301':
            lookup['Cz'] = rgi((alpha1,beta1,dele1),aerodata[file])    
        elif file == 'CM0120_ALPHA1_BETA1_DH1_101':
            lookup['Cm'] = rgi((alpha1,beta1,dele1),aerodata[file])
        elif file == 'CY0320_ALPHA1_BETA1_401':
            lookup['Cy'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0120_ALPHA1_BETA1_DH2_501':
            lookup['Cn'] = rgi((alpha1,beta1,dele2),aerodata[file])
        elif file == 'CL0120_ALPHA1_BETA1_DH2_601':
            lookup['Cl'] = rgi((alpha1,beta1,dele2),aerodata[file])


        elif file == 'CX0820_ALPHA2_BETA1_202':
            lookup['Cx_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CZ0820_ALPHA2_BETA1_302':
            lookup['Cz_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CM0820_ALPHA2_BETA1_102':
            lookup['Cm_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CY0820_ALPHA2_BETA1_402':
            lookup['Cy_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CN0820_ALPHA2_BETA1_502':
            lookup['Cn_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CL0820_ALPHA2_BETA1_602':
            lookup['Cl_lef'] = rgi((alpha2,beta1),aerodata[file])
        
        elif file == 'CX1120_ALPHA1_204':
            lookup['CXq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CZ1120_ALPHA1_304':
            lookup['CZq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CM1120_ALPHA1_104':
            lookup['CMq'] = interp1d(alpha1,aerodata[file])
        elif file == 'CY1220_ALPHA1_408':
            lookup['CYp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CY1320_ALPHA1_406':
            lookup['CYr'] = interp1d(alpha1,aerodata[file])
        elif file == 'CN1320_ALPHA1_506':
            lookup['CNr'] = interp1d(alpha1,aerodata[file])
        elif file == 'CN1220_ALPHA1_508':
            lookup['CNp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL1220_ALPHA1_608':
            lookup['CLp'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL1320_ALPHA1_606':
            lookup['CLr'] = interp1d(alpha1,aerodata[file])

        elif file == 'CX1420_ALPHA2_205':
            lookup['delta_CXq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CY1620_ALPHA2_407':
            lookup['delta_CYr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CY1520_ALPHA2_409':
            lookup['delta_CYp_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CZ1420_ALPHA2_305':
            lookup['delta_CZq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CL1620_ALPHA2_607':
            lookup['delta_CLr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CL1520_ALPHA2_609':
            lookup['delta_CLp_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CM1420_ALPHA2_105':
            lookup['delta_CMq_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CN1620_ALPHA2_507':
            lookup['delta_CNr_lef'] = interp1d(alpha2,aerodata[file])
        elif file == 'CN1520_ALPHA2_509':
            lookup['delta_CNp_lef'] = interp1d(alpha2,aerodata[file])

        elif file == 'CY0720_ALPHA1_BETA1_405':
            lookup['Cy_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0720_ALPHA1_BETA1_503':
            lookup['Cn_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CL0720_ALPHA1_BETA1_603':
            lookup['Cl_r30'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CY0620_ALPHA1_BETA1_403':
            lookup['Cy_a20'] = rgi((alpha1,beta1),aerodata[file])

        elif file == 'CY0920_ALPHA2_BETA1_404':
            lookup['Cy_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        
        elif file == 'CN0620_ALPHA1_BETA1_504':
            lookup['Cn_a20'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CN0920_ALPHA2_BETA1_505':
            lookup['Cn_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CL0620_ALPHA1_BETA1_604':
            lookup['Cl_a20'] = rgi((alpha1,beta1),aerodata[file])
        elif file == 'CL0920_ALPHA2_BETA1_605':
            lookup['Cl_a20_lef'] = rgi((alpha2,beta1),aerodata[file])
        elif file == 'CN9999_ALPHA1_brett':
            lookup['delta_CNbeta'] = interp1d(alpha1,aerodata[file])
        elif file == 'CL9999_ALPHA1_brett':
            lookup['delta_CLbeta'] = interp1d(alpha1,aerodata[file])
        elif file == 'CM9999_ALPHA1_brett':
            lookup['delta_Cm'] = interp1d(alpha1,aerodata[file])
        elif file == 'ETA_DH1_brett':
            lookup['eta_el'] = interp1d(dele1,aerodata[file])
    
    # hifi_C_lef
    lookup['delta_Cx_lef'] = lambda alpha, beta: lookup['Cx_lef']((alpha,beta)) - lookup['Cx']((alpha,beta,0))
    lookup['delta_Cz_lef'] = lambda alpha, beta: lookup['Cz_lef']((alpha,beta)) - lookup['Cz']((alpha,beta,0))
    lookup['delta_Cm_lef'] = lambda alpha, beta: lookup['Cm_lef']((alpha,beta)) - lookup['Cm']((alpha,beta,0))
    lookup['delta_Cy_lef'] = lambda alpha, beta: lookup['Cy_lef']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cn_lef'] = lambda alpha, beta: lookup['Cn_lef']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cl_lef'] = lambda alpha, beta: lookup['Cl_lef']((alpha,beta)) - lookup['Cl']((alpha,beta,0))

    # hifi_rudder
    lookup['delta_Cy_r30'] = lambda alpha, beta: lookup['Cy_r30']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cn_r30'] = lambda alpha, beta: lookup['Cn_r30']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cl_r30'] = lambda alpha, beta: lookup['Cl_r30']((alpha,beta)) - lookup['Cl']((alpha,beta,0))

    # hifi_ailerons
    lookup['delta_Cy_a20'] = lambda alpha, beta: lookup['Cy_a20']((alpha,beta)) - lookup['Cy']((alpha,beta))
    lookup['delta_Cy_a20_lef'] = lambda alpha, beta: lookup['Cy_a20_lef']((alpha,beta)) - lookup['Cy_lef']((alpha,beta)) - lookup['delta_Cy_a20'](alpha,beta)
    lookup['delta_Cn_a20'] = lambda alpha, beta: lookup['Cn_a20']((alpha,beta)) - lookup['Cn']((alpha,beta,0))
    lookup['delta_Cn_a20_lef'] = lambda alpha, beta: lookup['Cn_a20_lef']((alpha,beta)) - lookup['Cn_lef']((alpha,beta)) - lookup['delta_Cn_a20'](alpha,beta)
    lookup['delta_Cl_a20'] = lambda alpha, beta: lookup['Cl_a20']((alpha,beta)) - lookup['Cl']((alpha,beta,0))
    lookup['delta_Cl_a20_lef'] = lambda alpha, beta: lookup['Cl_a20_lef']((alpha,beta)) - lookup['Cl_lef']((alpha,beta)) - lookup['delta_Cl_a20'](alpha,beta)

    return lookup

def parse_aerodata(fp):

    '''
    As you might expect from its name this function parses the aerodynamic data
    found in the aerodata file. What you might not expect is that it will produce
    a nice pythonic dictionary for accessing these files.
    '''

    # isolate the aerodata directory and list the files
    aerodata_path = fp
    files = os.listdir(aerodata_path)

    # read in the raw .dat files
    aerodata_raw = {}
    for file in files:
        aerodata_raw[os.path.splitext(file)[0]] = open(aerodata_path + '/' + file)

    '''
    There are two sets of aerodynamic datapoints - broadly they can be thought
    of as high fidelity and low fidelity. There exists two sets of alpha and
    dele values. Sometimes they are also mixed.

    For example we could have a set of high fidelity alpha points with low
    fidelity dele points.

    Below I manually entered the number of possible points of alpha and beta
    and dele, and then created a dictionary 'ndinfo' containing the dimensionality
    information of every possible combination of them.
    '''
    ndinfo = {}
    ndinfo['alpha1_points'] = 20
    ndinfo['alpha2_points'] = 14
    ndinfo['beta1_points'] = 19
    ndinfo['dele1_points'] = 5
    ndinfo['dele2_points'] = 3

    # 3D tables
    # here a1b1d1 refers to using the alpha1 beta1 and dele1 point sets
    ndinfo['3D_a1b1d1_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points'], ndinfo['dele1_points']]
    ndinfo['3D_a1b1d1_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points'] * ndinfo['dele1_points']
    ndinfo['3D_a1b1d2_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points'], ndinfo['dele2_points']]
    ndinfo['3D_a1b1d2_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points'] * ndinfo['dele2_points']

    # 2D tables
    # similarly a1b1 refers to alpha1 and beta1 point sets
    ndinfo['2D_a1b1_shape'] = [ndinfo['alpha1_points'], ndinfo['beta1_points']]
    ndinfo['2D_a1b1_size'] = ndinfo['alpha1_points'] * ndinfo['beta1_points']
    ndinfo['2D_a2b1_shape'] = [ndinfo['alpha2_points'], ndinfo['beta1_points']]
    ndinfo['2D_a2b1_size'] = ndinfo['alpha2_points'] * ndinfo['beta1_points']
    ndinfo['2D_a1d1_shape'] = [ndinfo['alpha1_points'],  ndinfo['dele1_points']]
    ndinfo['2D_a1d1_size'] = ndinfo['alpha1_points'] * ndinfo['dele1_points']

    # 1D tables
    ndinfo['1D_a1_shape'] = ndinfo['alpha1_points']
    ndinfo['1D_a1_size'] = ndinfo['alpha1_points']
    ndinfo['1D_d1_shape'] = ndinfo['dele1_points']
    ndinfo['1D_d1_size'] = ndinfo['dele1_points']

    # instantiate dictionaries
    aerodata_list = {}
    aerodata = {}

    # reshape data into aerodata dictionary
    for i in aerodata_raw:
        # get all the dat files in 1D lists
        aerodata_list[i] = [float(x) for x in aerodata_raw[i].read().split()]
        # we need to reshape these 1D lists into 1D, 2D, or 3D tables
        # if its an alpha1, beta1 and dele1 table
        if len(aerodata_list[i]) == ndinfo['3D_a1b1d1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['3D_a1b1d1_shape'])
        # if its an alpha1, beta1 and dele2 table
        elif len(aerodata_list[i]) == ndinfo['3D_a1b1d2_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['3D_a1b1d2_shape'])
        # if its an alpha1 and beta1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a1b1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a1b1_shape'])
        # if its an alpha2 and beta1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a2b1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a2b1_shape'])

        # if its an alpha1 and dele1 table
        elif len(aerodata_list[i]) == ndinfo['2D_a1d1_size']:
            aerodata[i] = np.array(aerodata_list[i]).reshape(ndinfo['2D_a1d1_shape'])
        # if its a 1D table
        else:
            aerodata[i] = np.array(aerodata_list[i])

    return aerodata, ndinfo
