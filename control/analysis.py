import numpy as np
import scipy

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def discrete_gram(A,B,C):
    # source: https://uk.mathworks.com/help/control/ref/ss.gram.html;jsessionid=e2d01bb6a726a08d45c48257df5c
    # controllability gramiam, an nxn matrix, Wc ~ C @ C.T for discrete system (empirical finding not just here)
    Wc = scipy.linalg.solve_discrete_lyapunov(A,B@B.T)
    # observation gramiam, an mxm matrix
    Wo = scipy.linalg.solve_discrete_lyapunov(A,C.T@C)
    # recall that a system is stabilizable if and only if all unstable (and lightly damped) eigenvectors of A are in my controllable subspace
    # meaning they are in the column space of the controllability matrix. So I want to choose my actuators, B s.t. the unstable dynamics directions correspond to big singular vectors in the controllability matrix 
    return Wc, Wo

def ctrb(A,B):
    
    m = A.shape[0] # number of states
    n = B.shape[1] # number of inputs
    
    ctrb_mat = np.zeros((m,n*m))
    
    # [B AB A^2B ... A^(m-1)B]
    
    for i in range(m):
        
        ctrb_mat[:,n*i:n*(i+1)] = np.linalg.matrix_power(A,i) @ B
        
    return ctrb_mat

def is_ctrb(A,B):
    if np.linalg.matrix_rank(ctrb(A,B)) == A.shape[0]:
        return True
    elif np.linalg.matrix_rank(ctrb(A,B)) <= A.shape[0]:
        return False
    
def is_obsv(A,C):
    if np.linalg.matrix_rank(obsv(A,C)) == A.shape[0]:
        return True
    elif np.linalg.matrix_rank(obsv(A,C)) <= A.shape[0]:
        return False

