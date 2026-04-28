# EMG processing imports
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6
import AdaptiveEmbodiedControlSystems.LSTM as LSTM

# Control imports

# motor imports

# normal imports
import time
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import signal

# ======================= Global Parameters =======================
FS = 2000  # Sampling frequency for EMG
dt = 1 / FS  # Time step
THETA_MIN = np.deg2rad(0)  # Minimum angle in radians
THETA_MAX = np.deg2rad(140)  # Maximum angle in radians
THETA_RANGE = THETA_MAX - THETA_MIN  # Range of motion in radians
USER_NAME = 'VictorBNielsen'  # User name for EMG interpretation
TORQUE_MIN = -4.1  # Minimum torque for motors
TORQUE_MAX = 4.1  # Maximum torque for motors
MAX_VEL = np.pi

LSTM_PATH = "Outputs/models/LSTM/Windowed_LSTM.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_event = threading.Event()

#=================================================================

def EMG_Processing(qd_queue):
    # Initialize filters and interpreter
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(FS, lp_cutoff=2, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # Initialize LSTM model
    model = LSTM.LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=1, batch_first=True).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    # Optimization parameters
    q = 0
    v = 0
    optimized_angle = 0
    delta_q_prev = 0
    a_prev = 0
    dif_a_prev = 0

    # Initialize EMG sensors
    emg = DelsysEMG(channel_range=(0,1))
    emg.start()
    time.sleep(1.0)  # Allow some time for the EMG to start and gather data

    # Main processing loop
    while not stop_event.is_set():
        reading = emg.read()
        time_stamp = time.time()

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
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        b = 4.0 # File 2
        k=np.pi*10.0*2 # File 1
        optimized_angle, v, acc = optimizer_6(filtered_net_a, v, dt, optimized_angle, THETA_MIN, THETA_MAX, np.pi, b, k)

        # LSTM prediction
        with torch.no_grad():
            lstm_input = torch.tensor([[optimized_angle]], dtype=torch.float32).to(device)
            lstm_output = model(lstm_input)
            predicted_angle = lstm_output.item()
        
        try:
            qd_queue.put_nowait((predicted_angle, time_stamp))
        except queue.Full:
            qd_queue.get_nowait()
            qd_queue.put_nowait((predicted_angle, time_stamp))

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# ==========================================================================================================================
# =========================================== Handle cntrl+c stop event ====================================================
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

# ==========================================================================================================================
# ======================================================= Main function ====================================================

if __name__ == "__main__":
    # Desired position queue
    qd_queue = queue.Queue(maxsize=3) # or 5

    #TODO:Define parameters here if needed otherwise do it in global parameters section

    # TODO: Initialize motors here
    time.sleep(1.0)  # Allow some time for motor initialization

    emg_thread = threading.Thread(target=EMG_Processing, args=(qd_queue,))
    emg_thread.start()

    # TODO: Here you would add the code to read from qd_queue and send commands to the exoskeleton motors
    while not stop_event.is_set():
        pass

    # If we dont want to manually stop it we just have to call this after while loop
    # stop_event.set()

    emg_thread.join()


