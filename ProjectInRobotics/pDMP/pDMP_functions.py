"""
PERIODIC DYNAMIC MOVEMENT PRIMITIVES (pDMP)

This periodic DMP system has 3 modes:
    - LEARN: learns the DMP shape based on some input signal
    - UPDATE: updates the DMP shape based on some input signal 
    - REPEAT: repeats the existing DMP


AUTHOR: Luka Peternel
e-mail: l.peternel@tudelft.nl


REFERENCE:
L. Peternel, T. Noda, T. Petrič, A. Ude, J. Morimoto and J. Babič
Adaptive control of exoskeleton robots for periodic assistive behaviours based on EMG feedback minimisation,
PLOS One 11(2): e0148942, Feb 2016

"""

import numpy as np

class pDMP:
    # INITIALISATION
    def __init__(self, DOF, N, alpha, beta, lambd, dt):
    
        # settings
        self.DOF = DOF # degrees of freedom (number of DMPs)
        self.N = N # number of kernel functions per DMP
        self.alpha = alpha # DMP gain alpha
        self.beta = beta # DMP gain beta
        self.lambd = lambd # forgetting factor
        self.dt = dt # sample time
        
        # DMP learning variables
        self.f = np.zeros([self.DOF]) # shape function
        self.w = np.zeros([self.DOF,self.N]) # DMP weights
        self.c = np.zeros([self.N]) # centers of kernel functions
        self.P = np.ones([self.DOF,self.N]) # regression variable P
        self.r = np.ones([self.DOF]) # amplitude parameter
        self.g = np.zeros([self.DOF]) # DMP goal variable
        
        # DMP state variables
        self.y = np.zeros([self.DOF]) 
        self.z = np.zeros([self.DOF]) 

        # DMP phase and period
        self.phi = np.zeros([self.DOF]) 
        self.tau = np.zeros([self.DOF])
        
        # define centers of kernel functions
        spread = ( 2 * np.pi - 0.0 ) / N # distance between the kernels
        for i in range(self.N):
            self.c[i] = 0.5 * spread + spread * i
    
    # GET STATE
    def get_state(self):
        dy = self.z / self.tau
        return self.y, dy, self.tau, self.phi
    
    # SET DMP PHASE
    def set_phase(self, phi):
        self.phi = phi
    
    # SET DMP PERIOD
    def set_period(self, tau):
        self.tau = tau
    
    # SET DMP WEIGHTS
    def set_weights(self, DOF, w):
        self.w[DOF,:] = w
    
    # GET DMP WEIGHTS
    def get_weights(self, DOF):
        return self.w[DOF,:]
    
    # SET DMP KERNELS
    def set_kernels(self, c):
        self.c = c
    
    # GET DMP KERNELS
    def get_kernels(self):
        return self.c
    
    # LEARN MODE
    def learn(self, y, dy, ddy):
        f_d = np.zeros([self.DOF])
        psi = np.zeros([self.N])
        
        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0
            
            # desired shape
            f_d[i] = self.tau[i]**2 * ddy[i] - self.alpha * (self.beta * ( self.g[i] - y[i] ) - self.tau[i] * dy[i])
            
            # recursive least-squares regression
            for j in range(self.N):
                # update kernels and weights
                psi[j] = np.exp( 2.5 * self.N * ( np.cos( self.phi[i]- self.c[j] ) - 1 ) )
                P_new = ( self.P[i,j] - ( self.P[i,j]**2 * self.r[i]**2 ) / ( self.lambd / psi[j] + self.P[i,j] * self.r[i]**2 ) ) / self.lambd
                self.w[i,j] += psi[j] * P_new * self.r[i] * ( f_d[i] - self.w[i,j] * self.r[i] )
                self.P[i,j] = P_new
                
                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]
            
            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum
        
    # UPDATE MODE
    def update(self, U):
        psi = np.zeros([self.N])
        
        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0
            
            # recursive least-squares regression
            for j in range(self.N):
                # update kernels and weights
                psi[j] = np.exp( 2.5 * self.N * ( np.cos( self.phi[i] - self.c[j] ) - 1 ) )
                P_new = ( self.P[i,j] - ( self.P[i,j]**2 * self.r[i]**2 ) / ( self.lambd / psi[j] + self.P[i,j] * self.r[i]**2 ) ) / self.lambd
                self.w[i,j] += psi[j] * P_new * self.r[i] * U[i]
                self.P[i,j] = P_new
                
                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]
                
            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum
    
    # REPEAT MODE
    def repeat(self):
        psi = np.zeros([self.N])
        
        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0
            
            # recursive least-squares regression
            for j in range(self.N):
                psi[j] = np.exp( 2.5 * self.N * ( np.cos(self.phi[i] - self.c[j] ) - 1 ) )
                
                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]
            
            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum

    # INTEGRATION
    def integration(self):
        for i in range(self.DOF):
            dz = ( 1 / self.tau[i] ) * ( self.alpha * ( self.beta * ( self.g[i] - self.y[i] ) - self.z[i] ) + self.f[i] )
            dy = ( 1 / self.tau[i] ) * self.z[i]
            
            self.y[i] += dy * self.dt
            self.z[i] += dz * self.dt


"""
author: Victor Brønsholm Nielsen
"""

class dDMP:
    """
    DISCRETE DYNAMIC MOVEMENT PRIMITIVES (dDMP)

    Modes (same as rhythmic pDMP):
      - LEARN(y, dy, ddy): learns the DMP shape from a signal (supervised)
      - UPDATE(U): updates the DMP shape from a teaching signal (unsupervised)
      - REPEAT(): evaluates the current DMP without learning

    Differences vs rhythmic:
      - uses canonical state s ∈ (0,1] instead of phase phi
      - kernels are Gaussians in s
      - set amplitude r[i] = (g[i] - y0[i]) to get standard discrete DMP behavior
    """

    # INITIALISATION
    def __init__(self, DOF, N, alpha, beta, lambd, dt, alpha_s=4.0):
        # settings
        self.DOF = DOF              # number of DMPs (DoFs)
        self.N = N                  # number of kernel functions per DMP
        self.alpha = alpha          # DMP gain alpha (transformation)
        self.beta = beta            # DMP gain beta (transformation)
        self.lambd = lambd          # RLS forgetting factor
        self.dt = dt                # sample time
        self.alpha_s = alpha_s      # canonical gain (if you advance s internally)

        # learning variables
        self.f = np.zeros([self.DOF])             # shape function value
        self.w = np.zeros([self.DOF, self.N])     # weights
        self.c = np.zeros([self.N])               # kernel centers in s-space
        self.h = np.ones([self.N])                # kernel widths in s-space
        self.P = np.ones([self.DOF, self.N])      # RLS covariance (per-weight)
        self.r = np.ones([self.DOF])              # amplitude scale (set to g - y0)
        self.g = np.zeros([self.DOF])             # goal

        # state variables
        self.y  = np.zeros([self.DOF])
        self.z  = np.zeros([self.DOF])            # scaled velocity
        self.y0 = np.zeros([self.DOF])            # start position (for convenience)

        # canonical state and duration
        self.s   = np.ones([self.DOF])            # canonical state in (0,1]
        self.tau = np.ones([self.DOF])            # movement duration (per DoF if needed)

        # define Gaussian kernel centers/widths in s ∈ (0,1]
        # geometric spacing (denser near s=0) works well for discrete DMPs
        grid = np.linspace(0, 1, self.N)
        self.c = np.exp(-self.alpha_s * grid)     # maps to s-like distribution
        d = np.diff(np.r_[self.c, 0.0])
        # widths so neighboring kernels overlap; tiny eps avoids div by zero
        self.h = 1.0 / ((0.5*np.r_[d[:-1], d[-1]])**2 + 1e-6)

    # ---------- STATE/SETTERS/GETTERS ----------
    def get_state(self):
        dy = self.z / self.tau
        return self.y, dy, self.tau, self.s

    def set_phase(self, s):
        """Set canonical state s (you typically decrease it externally)."""
        self.s = s

    def set_period(self, tau):
        """Set movement duration tau (kept for API parity)."""
        self.tau = tau

    def set_start(self, y0):
        self.y0 = y0

    def set_goal(self, g):
        self.g = g

    def set_amplitude_from_start_goal(self):
        """Convenience: r = g - y0 (standard discrete DMP amplitude)."""
        self.r = self.g - self.y0

    def set_weights(self, DOF, w):
        self.w[DOF, :] = w

    def get_weights(self, DOF):
        return self.w[DOF, :]

    def set_kernels(self, c, h=None):
        self.c = c
        if h is not None:
            self.h = h

    def get_kernels(self):
        return self.c, self.h

    # ---------- INTERNAL: KERNEL ACTIVATIONS ----------
    def _psi(self, s_scalar):
        return np.exp(-self.h * (s_scalar - self.c)**2)

    # ---------- MODES ----------
    # LEARN MODE (supervised from y, dy, ddy)
    def learn(self, y, dy, ddy):
        f_d = np.zeros([self.DOF])

        for i in range(self.DOF):
            # desired forcing term for discrete DMP (scaled spring-damper residual)
            # f_d = tau^2*ydd - alpha*(beta*(g - y) - tau*yd)
            f_d[i] = self.tau[i]**2 * ddy[i] - self.alpha * (self.beta * (self.g[i] - y[i]) - self.tau[i] * dy[i])

            psi = self._psi(self.s[i])
            psi_sum = np.sum(psi)

            # RLS per basis (same structure as your periodic code)
            for j in range(self.N):
                Pij = self.P[i, j]
                rij = self.r[i]      # amplitude scale
                # guarded denominator (keeps identical algebraic shape)
                denom = (self.lambd / (psi[j] + 1e-12)) + Pij * (rij**2)
                P_new = (Pij - (Pij**2 * rij**2) / denom) / self.lambd
                # target here is f_d; model output ~ w*r
                self.w[i, j] += psi[j] * P_new * rij * (f_d[i] - self.w[i, j] * rij)
                self.P[i, j] = P_new

            # evaluate current forcing f (normalized weighted sum)
            if psi_sum == 0:
                self.f[i] = 0.0
            else:
                self.f[i] = (self.w[i, :] @ (psi * self.r[i])) / psi_sum

    # UPDATE MODE (unsupervised; push weights by teaching signal U)
    def update(self, U):
        for i in range(self.DOF):
            psi = self._psi(self.s[i])
            psi_sum = np.sum(psi)
            weighted_sum = 0.0

            for j in range(self.N):
                Pij = self.P[i, j]
                rij = self.r[i]
                denom = (self.lambd / (psi[j] + 1e-12)) + Pij * (rij**2)
                P_new = (Pij - (Pij**2 * rij**2) / denom) / self.lambd
                # teaching signal U[i] acts like 'increment to f' at this s
                self.w[i, j] += psi[j] * P_new * rij * U[i]
                self.P[i, j] = P_new
                weighted_sum += self.w[i, j] * psi[j] * rij

            self.f[i] = 0.0 if psi_sum == 0 else weighted_sum / psi_sum

    # REPEAT MODE (no learning; just evaluate f at current s)
    def repeat(self):
        for i in range(self.DOF):
            psi = self._psi(self.s[i])
            psi_sum = np.sum(psi)
            weighted_sum = self.w[i, :] @ (psi * self.r[i])
            self.f[i] = 0.0 if psi_sum == 0 else weighted_sum / psi_sum

    # ---------- INTEGRATION (same structure as your code) ----------
    def integration(self):
        for i in range(self.DOF):
            # discrete DMP dynamics (forcing already includes amplitude via r)
            dz = (1.0 / self.tau[i]) * ( self.alpha * ( self.beta * ( self.g[i] - self.y[i] ) - self.z[i] ) + self.f[i] )
            dy = (1.0 / self.tau[i]) * self.z[i]

            self.y[i] += dy * self.dt
            self.z[i] += dz * self.dt

    # ---------- OPTIONAL: INTERNAL CANONICAL ADVANCE (if you want it inside) ----------
    def step_canonical(self, a_s=None):
        """Advance s internally: ds = -a_s*s/tau * dt"""
        a = self.alpha_s if a_s is None else a_s
        self.s += (-a * self.s / self.tau) * self.dt
        self.s = np.maximum(self.s, 1e-6)
