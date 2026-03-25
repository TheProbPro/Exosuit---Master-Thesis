"""
PERIODIC DYNAMIC MOVEMENT PRIMITIVES (pDMP)

An example of how to use pDMP functions.


AUTHOR: Luka Peternel
e-mail: l.peternel@tudelft.nl


REFERENCE:
L. Peternel, T. Noda, T. Petrič, A. Ude, J. Morimoto and J. Babič
Adaptive control of exoskeleton robots for periodic assistive behaviours based on EMG feedback minimisation,
PLOS One 11(2): e0148942, Feb 2016

"""

import numpy as np
import matplotlib.pyplot as plt
from ProjectInRobotics.pDMP.pDMP_functions import pDMP
from ProjectInRobotics.pDMP.pDMP_functions import pDMPCoupling1
from ProjectInRobotics.pDMP.pDMP_functions import pDMPOmega


# EXPERIMENT PARAMETERS
dt = 1/166 # system sample time
exp_time = 10 # total experiment time
samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25#25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.9 # forgetting factor
tau = 0.5 #10 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = 0
dy_old = 0

data = []

# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)


# Experiment 1 - learn a 0 trajectory and modify the trajectory based on the update function with EMG signals.
v = (1.3*np.pi)/2
# MAIN LOOP
for i in range ( samples ):

    # generate phase
    phi += 16*np.pi * dt/tau
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([0])
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # set phase and period for DMPs
    myDMP.set_phase( np.array([phi]) )
    myDMP.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 0.2 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            myDMP.update(U) # update DMP based on an update factor
    elif i >= int( 0.2 * samples ) and i < int( 0.4 * samples ): # repeat the learned DMP for a while
        U = np.asarray([1 * v]) # example of a constant update factor (e.g., constant EMG signal)
        myDMP.update(U)
    elif i >= int( 0.4 * samples ) and i < int( 0.6 * samples ): # repeat the learned DMP for a while
        U = np.asarray([0]) # example of a zero update factor (e.g., no EMG signal)
        myDMP.update(U)
    elif i >= int( 0.6 * samples ) and i < int( 0.8 * samples ): # repeat the learned DMP until the end of the experiment
        U = np.asarray([-1 * v]) # example of a negative update factor (e.g., opposite EMG signal)
        myDMP.update(U)
    else : # repeat the learned DMP until the end of the experiment
        U = np.asarray([0])
        myDMP.update(U)
    
    # DMP integration
    myDMP.integration()
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = myDMP.get_state()
    time = dt*i
    data.append([time,phi,x[0],y[0]])

# PLOTS
data = np.asarray(data)

# input
plt.plot(data[:,0],data[:,3],'r')
# DMP
plt.plot(data[:,0],data[:,2],'b')

plt.xlabel('time [s]', fontsize='12')
plt.ylabel('signal', fontsize='13')

plt.legend(['input','DMP'])

plt.title('Periodic DMP', fontsize='14')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Experiement 3 - Learn a sinusoidal trajectory and modify the trajectory using the coupling term aka. the EMG activation signal.

# EXPERIMENT PARAMETERS
dt = 1/166 # system sample time
exp_time = 10 # total experiment time
samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25#25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.9 # forgetting factor
tau = 0.5 #10 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = 0
dy_old = 0

data = []

# create a DMP object
myDMP = pDMPCoupling1(DOF, N, alpha, beta, lambd, dt)

# MAIN LOOP
for i in range ( samples ):

    # generate phase
    phi += 2*np.pi * dt/tau
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    # y = np.asarray([np.sin(phi)])
    y = np.asarray([0])
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # set phase and period for DMPs
    myDMP.set_phase( np.array([phi]) )
    myDMP.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 0.2 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
    else:
        myDMP.repeat()
    
    # DMP integration
    if i >= int( 0.2 * samples ) and i < int( 0.49 * samples ): # repeat the learned DMP for a while
        myDMP.integration(np.asarray([1]))
    elif i >= int( 0.49 * samples ) and i < int( 0.6 * samples ): # repeat the learned DMP until the end of the experiment
        myDMP.integration(np.asarray([0]))
    elif i >= int( 0.6 * samples ) and i < int( 0.89 * samples ): # repeat the learned DMP until the end of the experiment
        myDMP.integration(np.asarray([-1]))
    else : # repeat the learned DMP until the end of the experiment
        myDMP.integration(np.asarray([0]))
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = myDMP.get_state()
    time = dt*i
    data.append([time,phi,x[0],y[0]])



# PLOTS
data = np.asarray(data)

# input
plt.plot(data[:,0],data[:,3],'r')
# DMP
plt.plot(data[:,0],data[:,2],'b')

plt.xlabel('time [s]', fontsize='12')
plt.ylabel('signal', fontsize='13')

plt.legend(['input','DMP'])

plt.title('Periodic DMP', fontsize='14')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

# # Experiment 4 - Learn a sinusoidal trajectory and modify the trajectory using the coupling term aka. the EMG activation signal.

# # MAIN LOOP
# for i in range ( samples ):

#     # generate phase
#     phi += 4*np.pi * dt/tau
    
#     # generate an example trajectory (e.g., the movement that is to be learned)
#     y = np.asarray([np.sin(phi)])
#     # calculate time derivatives
#     dy = (y - y_old) / dt 
#     ddy = (dy - dy_old) / dt
    
#     # set phase and period for DMPs
#     myDMP.set_phase( np.array([phi]) )
#     myDMP.set_period( np.array([tau]) )
    
#     # DMP mode of operation
#     if i < int( 0.2 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
#         myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
#     else:
#         myDMP.repeat()
    
#     # DMP integration
#     if i >= int( 0.2 * samples ) and i < int( 0.4 * samples ): # repeat the learned DMP for a while
#         myDMP.integration(1)
#     elif i >= int( 0.4 * samples ) and i < int( 0.6 * samples ): # repeat the learned DMP until the end of the experiment
#         myDMP.integration(0)
#     elif i >= int( 0.6 * samples ) and i < int( 0.8 * samples ): # repeat the learned DMP until the end of the experiment
#         myDMP.integration(-1)
#     else : # repeat the learned DMP until the end of the experiment
#         myDMP.integration(0)
    
#     # old values	
#     y_old = y
#     dy_old = dy
    
#     # store data for plotting
#     x, dx, ph, ta = myDMP.get_state()
#     time = dt*i
#     data.append([time,phi,x[0],y[0]])



# # PLOTS
# data = np.asarray(data)

# # input
# plt.plot(data[:,0],data[:,3],'r')
# # DMP
# plt.plot(data[:,0],data[:,2],'b')

# plt.xlabel('time [s]', fontsize='12')
# plt.ylabel('signal', fontsize='13')

# plt.legend(['input','DMP'])

# plt.title('Periodic DMP', fontsize='14')
# plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

# # Experiment 5 - Learn a sinusoidal trajectory and modify the trajectory using the omega term aka. the DMP frequency.

# EXPERIMENT PARAMETERS
dt = 0.05#1/166 # system sample time
exp_time = 20#100 # total experiment time
samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25#25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.9 # forgetting factor
tau = 5 #10 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = 0
dy_old = 0

data = []

# create a DMP object
myDMP = pDMPOmega(DOF, N, alpha, beta, lambd, dt)

omega0 = 2 * np.pi / tau
k = 1.0 #0.9
myDMP.set_frequency(np.array([omega0]))
for i in range(samples):

    t = i * dt

    # generate demo trajectory
    y = np.asarray([np.sin(omega0 * t)])

    dy = (y - y_old) / dt
    ddy = (dy - dy_old) / dt

    # learning phase
    if i < int(0.2 * samples):
        myDMP.set_frequency(np.array([omega0]))
        myDMP.learn(y, dy, ddy)

    # control omega
    if i >= int(0.2 * samples) and i < int(0.4 * samples):
        omega = omega0 * (1 + k * 1)
        myDMP.set_frequency(np.array([omega]))
    elif i >= int(0.4 * samples) and i < int(0.6 * samples):
        omega = omega0 * (1 + k * 0)
        myDMP.set_frequency(np.array([omega]))
    elif i >= int(0.6 * samples) and i < int(0.8 * samples):
        omega = omega0 * (1 + k * -1)
        myDMP.set_frequency(np.array([omega]))
    else:
        omega = omega0 * (1 + k * 0)
        myDMP.set_frequency(np.array([omega]))

    # integrate (this updates phi internally)
    myDMP.repeat() # update phi for repeating the DMP
    myDMP.integration()

    # update old values
    y_old = y
    dy_old = dy

    # store data
    x, dx, phi, Omega = myDMP.get_state()
    data.append([t, x[0], y[0], dx[0], Omega[0]])

data = np.asarray(data)

# input vs DMP
plt.figure()
plt.plot(data[:,0], data[:,2], 'r', label='input')
plt.plot(data[:,0], data[:,1], 'b', label='DMP')

plt.xlabel('time [s]')
plt.ylabel('signal')
plt.title('Periodic DMP (Omega control)')
plt.legend()
plt.grid()
plt.show()

