'''
The EMG processing tests will consist of both prediction and optimization algorithms.
The prediction algorithms should be tested both before and after the optimization algorithms, and both at the same time.
So this means:
 EMG -> optimization -> prediction
 EMG -> prediction -> optimization
 EMG -> prediction -> optimization -> prediction
'''
# Custom includes
from Sensors.EMGSensor import DelsysEMG, DelsysEMGIMU
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_3, optimize_4, optimize_5_pd
from AdaptiveEmbodiedControlSystems.ESN import ESN
from AdaptiveEmbodiedControlSystems.LSTM import LSTM

# TODO: add includes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import queue
import time
import signal
import math
from ahrs.filters import Madgwick

'''
Tests: firstly just EMG no IMU, then test the best performing ones with both EMG and IMU.
'''

# Define global parameters
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

EMG_FS = 2000  # EMG sampling frequency (Hz)
MOTOR_FS = 166.7  # Motor control frequency (Hz)
IMU_FS = 0  # IMU sampling frequency (Hz) TODO: SET THIS LATER

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

stop_event = threading.Event()

# Define threading method for gathering emg data
def read_EMG(emg_pos_queue, emg_activation_queue):
    # Initialize filters
    filter_bicep = rt_filtering(EMG_FS, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USERNAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    time.sleep(1.0)
    
    while not stop_event.is_set():
        reading = emg.read()

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])

        try:
            emg_activation_queue.put_nowait(activation)
        except queue.Full:
            emg_activation_queue.get_nowait()
            emg_activation_queue.put_nowait(activation)

        desired_angle_deg = math.degrees(interpreter.compute_angle(activation[0], activation[1]))

        try:
            emg_pos_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            emg_pos_queue.get_nowait()
            emg_pos_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# Define main
if __name__ == "__main__":
    # Define EMG queues
    emg_pos_queue = queue.Queue(maxsize=5)
    emg_activation_queue = queue.Queue(maxsize=5)

    # Define desired angle lowpass filter
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=EMG_FS, lp_cutoff=3, order=2)
