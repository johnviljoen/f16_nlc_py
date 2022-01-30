import numpy as np
import scipy

# In[ discrete linear quadratic regulator ]
# from https://github.com/python-control/python-control/issues/359:
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    
    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*(x_ref[k] - x[k])
    select the states that you want considered and make x[k] the difference
    between the current x and the desired x.
      
    How to apply the function:    
        K = dlqr(A_d,B_d,Q,R)
      
    Inputs:
      A_d, B_d, Q, R  -> all numpy arrays  (simple float number not allowed)
      
    Returns:
      K: state feedback gain
    
    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ P @ B+R) @ (B.T @ P @ A))
    return K