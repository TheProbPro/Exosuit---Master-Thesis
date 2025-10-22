import numpy as np
import pandas as pd
import os

class ProportionalMyoelectricalControl:
    def __init__(self, theta_min, theta_max, user_name='User', BicepEMG = True, TricepEMG = False):
        """
        Initialize the Proportional Myoelectrical Control class.
        :param theta_min: Minimum angle (radians) (this can maybe be raw motor position value)
        :param theta_max: Maximum angle (radians) (this can maybe be raw motor position value)
        :param user_name: Name of the user for calibration data storage
        :param BicepEMG: Boolean to indicate if bicep EMG is used
        :param TricepEMG: Boolean to indicate if tricep EMG is used
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.user_name = user_name
        self.Bicep_EMG = BicepEMG
        self.Tricep_EMG = TricepEMG
        self.Kp = 1.0  # Proportional gain for torque computation

        # Check if there is a user with given name
        if not os.path.exists(f'Calib/Users/{self.user_name}'):
            print("Please run the calibration script for the user before starting the program!")
            raise ValueError("User not found!")

        # Load users biscep and tricep rest signal from .csv file
        df = pd.read_csv(f'Calib/Users/{self.user_name}/rest_signal.csv')
        if BicepEMG:
            self.bicep_rest = float(df['Bicep'])
        if TricepEMG:
            self.tricep_rest = float(df['Tricep'])
        
        # Load users biscep and tricep max signal from .csv file
        df = pd.read_csv(f'Calib/Users/{self.user_name}/max_signal.csv')
        if BicepEMG:
            self.bicep_max = float(df['Bicep'])
        if TricepEMG:
            self.tricep_max = float(df['Tricep'])

    def compute_activation(self, env):
        if not isinstance(env, (np.ndarray, list)):
            if self.Bicep_EMG:
                env = [env, 0.0]
            elif self.Tricep_EMG:
                env = [0.0, env]
            else:
                raise ValueError("At least BicepEMG or TricepEMG must be True")
        a_bicep = 0
        a_tricep = 0
        if self.Bicep_EMG:
            a_bicep = np.clip(((env[0] - self.bicep_rest) / (self.bicep_max - self.bicep_rest)), 0, 1)
        if self.Tricep_EMG:
            a_tricep = np.clip(((env[1] - self.tricep_rest) / (self.tricep_max - self.tricep_rest)), 0, 1)

        return a_bicep, a_tricep
    
    def _deadband(self, x, eps):
        if abs(x) < eps:
            return 0.0
        return x - eps * (1 if x > 0 else -1)

    def compute_angle(self, Bicep_activation, Tricep_activation):
        theta = 0
        if self.Bicep_EMG and self.Tricep_EMG:
            activation = (Bicep_activation - Tricep_activation)
            theta = (self.theta_min + self.theta_max)/2 + (self.theta_max - self.theta_min)/2 * self._deadband(activation, 0.1) #TODO: Tune eps (threshold value to avoid small activations from noise)
        elif self.Bicep_EMG:
            activation = Bicep_activation
            theta = self.theta_min + activation * (self.theta_max - self.theta_min)
        else:
            raise ValueError("At least BicepEMG or BicepEMG and TricepEMG must be True")

        return theta

    def set_Kp(self, Kp):
        self.Kp = Kp
    
    def compute_torque(self, env):
        torque = env * self.Kp
        return torque

    

class BiomechanicalMyoelectricalControl:
    def __init__(self):
        pass