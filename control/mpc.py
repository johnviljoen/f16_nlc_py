import numpy as np

def setup_OSQP(x_ref, A, B, Q, R, hzn, dt, x, act_states, x_lb, x_ub, u_lb, u_ub, udot_lb, udot_ub):
    
    """
    Function that builds a model predictive control problem from a discrete linear
    state space system (A,B), its corresponding cost function weights (Q,R), and
    its constraints. This is a tracking controller (x_ref) that aims for zero error
    in p,q,r angular rates.
    
    The output is designed for the OSQP constrained quadratic solver (https://osqp.org/docs/)
    
    The function is designed to eventually be generalisable with the rest of the 
    code, but this is not yet complete. This is the reason for some of the unusual
    decisions at first glance, like some input vectors being horizontal 1D and others
    vertical 2D.
    
    args:
        xref:
            1D numpy array -> horizontal vector (0 x m*hzn)
        A:
            2D numpy array (m x m)
        B:
            2D numpy array (m x n)
        Q:
            2D numpy array (m x m)
        R:
            2D numpy array (n x n)
        hzn:
            int
        dt:
            float
        x:
            1D numpy array -> horizontal vector (0 x m)
        act_states:
            1D numpy array -> horizontal vector (0 x n)
        x_lb:
            2D numpy array -> vertical vector (m x 1)
        x_ub:
            2D numpy array -> vertical vector (m x 1)
        u_lb:
            2D numpy array -> vertical vector (n x 1)
        u_ub:
            2D numpy array -> vertical vector (n x 1)      
        udot_lb:
            2D numpy array -> vertical vector (n x 1)
        udot_ub:
            2D numpy array -> vertical vector (n x 1)
            
    returns:
        OSQP_P:
            2D numpy array (n*hzn x n*hzn)
        OSQP_q:
            2D numpy array -> vertical vector (n*hzn x 1)
        OSQP_A:
            2D numpy array (m*hzn x n*hzn)
        OSQP_l:
            2D numpy array -> vertical vector (n*hzn x 1)
        OSQP_u:
            2D numpy array -> vertical vector (n*hzn x 1)
            
    """
    
    m = len(x)                      # number of states
    n = len(act_states)             # number of inputs
    
    x = x[:,None]                   # convert x to vertical vector
    
    x_ref = np.tile(x_ref, hzn)     # stack x_refs to create sequence
    x_ref = x_ref[:,None]     # convert stacked x_ref to vertical vector
    
    # calculate matrices for predictions (p16 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    MM, CC = calc_MC(A, B, hzn)
    
    # calculate LQR gain matrix for mode 2 (https://github.com/python-control/python-control/issues/359)
    
    K = - dlqr(A, B, Q, R)
    
    # calculate terminal weighting matrix (p24 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    Q_bar = scipy.linalg.solve_discrete_lyapunov((A + B @ K).T, Q + K.T @ R @ K)
        
    # construct full QQ, RR (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    
    QQ = dmom(Q, hzn)
    QQ[-m:,-m:] = Q_bar
    RR = dmom(R, hzn)
    
    # construct objective function (2.3) (p17 https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf)
    # and implement this in OSQP format
    
    OSQP_P = 2 * (CC.T @ QQ @ CC + RR)
    OSQP_q = -2 * ((x_ref - MM @ x).T @ QQ @ CC).T
    
    """
    There are three constraints to be enforced on the system:
        
        state constraints:
            x(n+1) = Ax(n) + Bu(n)
            
        input command limit constraints:
            u_min <= u <= u_max
            
        input command rate limit constraints:
            udot_min <= udot <= udot_max
    """
    
    # calculate state constraint limits vector
        
    x_lb = np.tile(x_lb,(hzn,1))    
    x_ub = np.tile(x_ub,(hzn,1))
    
    state_constr_lower = x_lb - MM @ x
    state_constr_upper = x_ub - MM @ x
    
    # the state constraint input sequence matrix is just CC
    
    # calculate the command saturation limits vector
    
    cmd_constr_lower = np.tile(u_lb,(hzn,1))
    cmd_constr_upper = np.tile(u_ub,(hzn,1))
    
    # calculate the command saturation input sequence matrix -> just eye
    
    cmd_constr_mat = np.eye(n*hzn)
    
    # calculate the command rate saturation limits vector
    
    u0_rate_constr_lower = act_states[:,None] + udot_lb * dt
    u0_rate_constr_upper = act_states[:,None] + udot_ub * dt
    
    cmd_rate_constr_lower = np.concatenate((u0_rate_constr_lower,np.tile(udot_lb,(hzn-1,1))))
    cmd_rate_constr_upper = np.concatenate((u0_rate_constr_upper,np.tile(udot_ub,(hzn-1,1))))
    
    # calculate the command rate saturation input sequence matrix
    
    cmd_rate_constr_mat = np.eye(n*hzn)
    for i in range(n*hzn):
        if i >= n:
            cmd_rate_constr_mat[i,i-n] = -1
            
    # assemble the complete matrices to send to OSQP
            
    OSQP_A = np.concatenate((CC, cmd_constr_mat, cmd_rate_constr_mat), axis=0)
    OSQP_l = np.concatenate((state_constr_lower, cmd_constr_lower, cmd_rate_constr_lower))
    OSQP_u = np.concatenate((state_constr_upper, cmd_constr_upper, cmd_rate_constr_upper))
    
    return OSQP_P, OSQP_q, OSQP_A, OSQP_l, OSQP_u    

# In[ Calculate the MM, CC matrices -> squiggly M and squiggly C in the notes ]

def calc_MC(A, B, hzn, includeFirstRow=False):
    
    # hzn is the horizon
    nstates = A.shape[0]
    ninputs = B.shape[1]
    
    # x0 is the initial state vector of shape (nstates, 1)
    # u is the matrix of input vectors over the course of the prediction of shape (ninputs,horizon)
    
    # initialise CC, MM, Bz
    CC = np.zeros([nstates*hzn, ninputs*hzn])
    MM = np.zeros([nstates*hzn, nstates])
    Bz = np.zeros([nstates, ninputs])
        
    for i in range(hzn):
        MM[nstates*i:nstates*(i+1),:] = np.linalg.matrix_power(A,i+1) 
        for j in range(hzn):
            if i-j >= 0:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = np.matmul(np.linalg.matrix_power(A,(i-j)),B)
            else:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = Bz
                
    if includeFirstRow:
        CC = np.concatenate((np.zeros([nstates,ninputs*hzn]), CC), axis=0)
        MM = np.concatenate((np.eye(nstates), MM), axis=0)

    return MM, CC

def calc_HFG(A, B, QQ, RR):
    
    hzn = np.rint((RR.shape[1]/B.shape[1])).astype(np.int64)
    
    
    MM, CC = calc_MC(A,B,hzn)
        
    H = CC.T @ QQ @ CC + RR
    F = CC.T @ QQ @ MM
    G = MM.T @ QQ @ MM
    
    return H, F, G

def calc_QR(n, C, hzn, Qbar_eq_Q=True):
    
    Q = C.T @ C
    R = np.eye(n) * 0.01
    
    QQ = dmom(Q, hzn)
    
    if C.ndim == 1:
        
        m = 1
        
    else:
    
        m = C.shape[1]
    
    if Qbar_eq_Q:
        
        QQ[-m:,-m:] = Q
    RR = dmom(R, hzn)
    
    return QQ, RR

def calc_L(H,F,n,hzn):
    
    pre_mult = np.zeros([n,n*hzn])
    pre_mult[0:n,0:n] = np.eye(n)
    
    return - pre_mult @ np.linalg.inv(H) @ F

