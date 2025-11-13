import numpy as np
import matplotlib.pyplot as plt
import queue
import threading
import sys
import time

#==============================================================================
# This code tests the emg part of the code
#==============================================================================

# My local imports (EMG sensor, filtering, interpretors, OIAC)
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

import signal

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

stop_event = threading.Event()

def read_EMG(EMG_sensor, queue):
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            queue.put_nowait(reading)
        except queue.Full:
            try:
                queue.get_nowait()  # Discard oldest data
                queue.put_nowait(reading)
            except queue.Full:
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)


# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create EMG sensor instance and setup thread
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    emg = DelsysEMG(channel_range=(0,16))

    # Initialize filtering and interpretors
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    interpreter.set_Kp(8)

    emg.start()
    # Start EMG reading thread
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_emg.start()
    print("EMG reading thread started!")

    # Filter and interpret the raw data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    joint_torque = 0.0
    last_position = 0
    i = 0
    while not stop_event.is_set():
        # Use a blocking get with timeout to avoid busy-waiting and to
        # allow the reader thread to drive the queue at its own rate.
        try:
            reading = raw_data.get_nowait()
        except queue.Empty:
            # no new raw data; yield CPU briefly and continue
            time.sleep(0.001)
            continue

        # Filter data
        filtered_Bicep = filter_bicep.bandpass(reading[0])
        filtered_Tricep = filter_tricep.bandpass(reading[1])

        # Calculate RMS
        try:
            if Bicep_RMS_queue.full():
                Bicep_RMS_queue.get_nowait()
            Bicep_RMS_queue.put_nowait(filtered_Bicep)
                
            if Tricep_RMS_queue.full():
                Tricep_RMS_queue.get_nowait()
            Tricep_RMS_queue.put_nowait(filtered_Tricep)
        except queue.Full:
            pass
            
        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))
            
        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_bicep_RMS = filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))
        filtered_tricep_RMS = filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))

        # Compute activation and joint torque
        activation = interpreter.compute_activation([filtered_bicep_RMS, filtered_tricep_RMS])
        #joint_torque = interpreter.compute_torque(activation)
        #print(f"Joint torque: {int(joint_torque) - 4}")
        position = interpreter.compute_angle(activation[0], activation[1])
        
        step = 1500/140
        step_offset = 1050
        position = 2550 - int(position*step)


    # Stop EMG reading thread and EMG sensor
    print("Shutting down...")
    stop_event.set()
    t_emg.join()
    emg.stop()
    # empty all queues
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    print("Goodbye!")
