import numpy as np

# degenerate a square 2d matrix
def square_mat_degen_2d(mat, degen_idx):
    
    degen_mat = np.zeros([len(degen_idx),len(degen_idx)])
    
    for i in range(len(degen_idx)):
        
        degen_mat[:,i] = mat[degen_idx, [degen_idx[i] for x in range(len(degen_idx))]]
        
    return degen_mat

# diagonal matrix of matrices
def dmom(mat, num_mats):
    '''
    Function to create a diagonal matrix of matrices (dmom) as is shown by my ascii 'art'

    1,2,3       1,2,3,0,0,0
    4,5,6   ->  4,5,6,0,0,0
    7,8,9       7,8,9,0,0,0
                0,0,0,1,2,3
                0,0,0,4,5,6
                0,0,0,7,8,9
    '''
    if isinstance(mat, np.ndarray):
    
        # dimension extraction
        nrows = mat.shape[0]
        ncols = mat.shape[1]
        
        # matrix of matrices matomats -> I thought it sounded cool
        matomats = np.zeros((nrows*num_mats,ncols*num_mats))
        
        for i in range(num_mats):
            for j in range(num_mats):
                if i == j:
                    matomats[nrows*i:nrows*(i+1),ncols*j:ncols*(j+1)] = mat
                    
    else:
        
        matomats = np.eye(num_mats) * mat
                
    return matomats