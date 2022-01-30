import matplotlib.pyplot as plt
from numpy import pi
# In[ visualise full 18 DoF system time history ]

def vis_mpc_u(u_storage, rng):
    fig, axs = plt.subplots(3,1)
    
    # axs[0].plot(rng, u_storage[:,0])
    # axs[0].set_ylabel('T_cmd')
    
    axs[0].plot(rng, u_storage[:,0])
    axs[0].set_ylabel('dh_cmd')
    
    axs[1].plot(rng, u_storage[:,1])
    axs[1].set_ylabel('da_cmd')
    
    axs[2].plot(rng, u_storage[:,2])
    axs[2].set_ylabel('dr_cmd')
    
def vis_mpc_x(x_storage, rng):
    
    fig1, axs1 = plt.subplots(2, 1)
    #fig.suptitle('Vertically stacked subplots')
        
    axs1[0].plot(rng, x_storage[:,0])
    axs1[0].set_ylabel('phi (rad)')
    
    axs1[1].plot(rng, x_storage[:,1])
    axs1[1].set_ylabel('theta (rad)')
    
    fig2, axs2 = plt.subplots(2,1)
    
    axs2[0].plot(rng, x_storage[:,2]*180/pi)
    axs2[0].set_ylabel('alpha (deg)')
    
    axs2[1].plot(rng, x_storage[:,3]*180/pi)
    axs2[1].set_ylabel('beta (deg)')
    
    fig3, axs3 = plt.subplots(3,1)
    
    axs3[0].plot(rng, x_storage[:,4]*180/pi)
    axs3[0].set_ylabel('p (deg/s)')
    
    axs3[1].plot(rng, x_storage[:,5]*180/pi)
    axs3[1].set_ylabel('q (deg/s)')
    
    axs3[2].plot(rng, x_storage[:,6]*180/pi)
    axs3[2].set_ylabel('r (deg/s)')
    axs3[2].set_xlabel('time (s)')

def vis_u(u_storage, rng):
    
    fig, axs = plt.subplots(3,1)
    
    # axs[0].plot(rng, u_storage[:,0])
    # axs[0].set_ylabel('T_cmd')
    
    axs[0].plot(rng, u_storage[:,0])
    axs[0].set_ylabel('dh_cmd')
    
    axs[1].plot(rng, u_storage[:,1])
    axs[1].set_ylabel('da_cmd')
    
    axs[2].plot(rng, u_storage[:,2])
    axs[2].set_ylabel('dr_cmd')
    
def vis_x(x_storage, rng):

    fig, axs = plt.subplots(12, 1)
    #fig.suptitle('Vertically stacked subplots')
    axs[0].plot(rng, x_storage[:,0])
    axs[0].set_ylabel('npos (ft)')
    
    axs[1].plot(rng, x_storage[:,1])
    axs[1].set_ylabel('epos (ft)')
    
    axs[2].plot(rng, x_storage[:,2])
    axs[2].set_ylabel('h (ft)')
    
    axs[3].plot(rng, x_storage[:,3])
    axs[3].set_ylabel('$\phi$ (rad)')
    
    axs[4].plot(rng, x_storage[:,4])
    axs[4].set_ylabel('$\theta$ (rad)')
    
    axs[5].plot(rng, x_storage[:,5])
    axs[5].set_ylabel('$\psi$ (rad)')
    
    axs[6].plot(rng, x_storage[:,6])
    axs[6].set_ylabel("V_t (ft/s)")
    
    axs[7].plot(rng, x_storage[:,7]*180/pi)
    axs[7].set_ylabel('alpha (deg)')
    
    axs[8].plot(rng, x_storage[:,8]*180/pi)
    axs[8].set_ylabel('beta (deg)')
    
    axs[9].plot(rng, x_storage[:,9]*180/pi)
    axs[9].set_ylabel('p (deg/s)')
    
    axs[10].plot(rng, x_storage[:,10]*180/pi)
    axs[10].set_ylabel('q (deg/s)')
    
    axs[11].plot(rng, x_storage[:,11]*180/pi)
    axs[11].set_ylabel('r (deg/s)')
    axs[11].set_xlabel('time (s)')
    
    fig2, axs2 = plt.subplots(5,1)
    
    axs2[0].plot(rng, x_storage[:,12])
    axs2[0].set_ylabel('P3')
    
    axs2[1].plot(rng, x_storage[:,13])
    axs2[1].set_ylabel('dh')
    
    axs2[2].plot(rng, x_storage[:,14])
    axs2[2].set_ylabel('da')
    
    axs2[3].plot(rng, x_storage[:,15])
    axs2[3].set_ylabel('dr')
    
    axs2[4].plot(rng, x_storage[:,16])
    axs2[4].set_ylabel('lef')
