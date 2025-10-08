import numpy as np

class EMGCostFunction:
    def __init__(self, T_Diag, sigma = 0.5):
        """
        T_Diag : list/ndarry of per-channel weights for the quadratic (>=0)¨
        sigma  : shape parameter; smaller -> cost rises faster with EMG 
        """
        self.T = np.diag(T_Diag)
        self.sigma2 = float(sigma) ** 2

    def costFunction(self, emg):
        """
        emg : current EMG envelope vector (per channel)
        Returns scalar cost in [0, 1].
        """
        q = emg.T @ self.T @ emg
        cost = 1 - np.exp(-(1/(2*self.sigma2)) * q)

        return cost
