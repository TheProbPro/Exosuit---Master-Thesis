import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering

# TODO: Test and make sure we can read and seperate Bicep and Tricep channels correctly

class ProportionalMyoelectricalControl:
    def __init__(self, theta_min, theta_max, user_name='User', BicepEMG = True, TricepEMG = False, DelsysEMG = None):
        """
        Initialize the Proportional Myoelectrical Control class.
        :param theta_min: Minimum angle (radians) (this can maybe be raw motor position value)
        :param theta_max: Maximum angle (radians) (this can maybe be raw motor position value)
        :param user_name: Name of the user for calibration data storage
        :param BicepEMG: Boolean to indicate if bicep EMG is used
        :param TricepEMG: Boolean to indicate if tricep EMG is used
        :param DelsysEMG: Instance of DelsysEMG class for reading EMG data
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.user_name = user_name
        self.Bicep_EMG = BicepEMG
        self.Tricep_EMG = TricepEMG

        # Check if there is a user with given name
        if not os.path.exists(f'Calib/Users/{self.user_name}'):
            os.makedirs(f'Calib/Users/{self.user_name}')
            print(f"New user created: {self.user_name}")
            self.calibrate(self.user_name, DelsysEMG)

        # Load users biscep and tricep rest signal from .csv file
        df = pd.read_csv(f'Calib/Users/{self.user_name}/rest_signal.csv')
        if BicepEMG:
            self.bicep_rest = df['Bicep']
        if TricepEMG:
            self.tricep_rest = df['Tricep']
        
        # Load users biscep and tricep max signal from .csv file
        df = pd.read_csv(f'Calib/Users/{self.user_name}/max_signal.csv')
        if BicepEMG:
            self.bicep_max = df['Bicep']
        if TricepEMG:
            self.tricep_max = df['Tricep']

    def calibrate(self, user_name,DelsysEMG = None):
        if DelsysEMG is None:
            raise ValueError("DelsysEMG instance must be provided for calibration")
        
        print("Starting calibration...")
        input("Press Enter and keep your arm relaxed for 10 seconds...")
        rest_data = []
        TIME = time.time()
        while (time.time() - TIME < 10):
            reading = DelsysEMG.read()
            rest_data.append(reading)
        
        # TODO: Maybe need to seperate between bicep and tricep channels here
        rt_filtering_instance = rt_filtering(2000, 300, 20, 2)
        filtered_rest = []
        chunk_size = 50
        for i in range(0, len(rest_data), chunk_size):
            chunk = rest_data[i:i+chunk_size]
            if len(chunk) < chunk_size:
                break
            _, filtered_chunk = rt_filtering_instance.process_chunk(chunk)
            filtered_rest.extend(filtered_chunk)
        
        filtered_rest = np.array(filtered_rest)
        mean_rest = np.mean(filtered_rest, axis=0)
        rest_data.clear()
        print("Rest signal recorded.")
        
        # Record MVC
        # TODO: Again seperate between bicep and tricep channels
        MVC_trials = 3
        all_max_data = []
        for trial in range(MVC_trials):
            input(f"Press Enter and perform Bicep Maximum Voluntary Contraction (MVC) for 5 seconds... Trial {trial+1}/{MVC_trials}")
            max_data = []
            TIME = time.time()
            while (time.time() - TIME < 5):
                reading = DelsysEMG.read()
                max_data.append(reading)
            all_max_data.append(max_data)
            print(f"MVC Trial {trial+1} recorded.")
        
        # filter the MVC attempts
        all_filtered_MVC = []
        for trial in range(MVC_trials):
            trial_data = []
            for i in range(len(all_max_data[trial])):
                chunk = all_max_data[trial][i:i+chunk_size]
                if len(chunk) < chunk_size:
                    break
                _, filtered_chunk = rt_filtering_instance.process_chunk(chunk)
                trial_data.extend(filtered_chunk)
            all_filtered_MVC.append(np.array(trial_data))
        all_filtered_MVC = np.array(all_filtered_MVC)
        MVC = []
        for filtered_MVC in all_filtered_MVC:
            MVC.append(self._calc_MVC(filtered_MVC))
        MVC = np.average(MVC, axis=0)
        all_max_data.clear()
        print("Max signal recorded.")

        # Save the Rest and MVC signals to .csv Files
        df_rest = pd.DataFrame()
        df_rest['Bicep'] = mean_rest
        df_rest['Tricep'] = mean_rest
        df_rest.to_csv(f'Calib/Users/{user_name}/rest_signal.csv', index=False)

        df_max = pd.DataFrame()
        df_max['Bicep'] = MVC
        df_max['Tricep'] = MVC
        df_max.to_csv(f'Calib/Users/{user_name}/max_signal.csv', index=False)
        print("Calibration data saved.")

    def _calc_MVC(self, signal, sampling_rate=2000, win_ms=800):
        i = np.max(signal)
        w = int(win_ms / 1000 * sampling_rate) // 2
        return np.mean(signal[max(0,i-w): i+w+1])


    def compute_activation(self, env):
        a_bicep = 0
        a_tricep = 0
        if self.Bicep_EMG:
            a_bicep = np.clip(((env - self.bicep_rest) / (self.bicep_max - self.bicep_rest)), 0, 1)
        if self.Tricep_EMG:
            a_tricep = np.clip(((env - self.tricep_rest) / (self.tricep_max - self.tricep_rest)), 0, 1)

        return a_bicep, a_tricep
    
    def _deadband(x, eps):
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
    

class BiomechanicalMyoelectricalControl:
    def __init__(self):
        pass