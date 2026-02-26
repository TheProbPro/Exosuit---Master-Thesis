# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# General imports
import numpy as np
import threading
import signal
import time
import matplotlib.pyplot as plt
import queue
import pandas as pd
import torch

import traceback

# General configuration parameters
SAMPLE_RATE = 135 #166.7  # Hz TODO: This needs to be measured again, since baudrate has been updated to 4.5Mbps
EMG_SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'

# TODO: This needs to be changed in accordance to finger limits
ANGLE_MIN = math.radians(0)
ANGLE_MAX = math.radians(140)

# TODO: Exosuit motor can apply torques of up to 10.6 Nm, but we limit it temporarely for safety
TORQUE_MAX = 4.1
TORQUE_MIN = -TORQUE_MAX

stop_event = threading.Event()

def read_EMG(raw_queue):
    """EMG读取线程"""
    # Initialize filters
    filter_bicep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
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

        raw_queue.put(activation)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Load MOTOR_POS min and max from calibration file if available
    df = pd.read_csv(f'Calib/Users/{USER_NAME}/finger_motor_calib_data.csv')
    MOTOR_POS_MIN = df['Extended'][0]
    MOTOR_POS_MAX = df['Flexed'][0]
    print(f"Loaded motor calibration data: MOTOR_POS_MIN={MOTOR_POS_MIN}, MOTOR_POS_MAX={MOTOR_POS_MAX}")

    EMG_queue = queue.Queue(maxsize=5)

    motor = Motors(port="COM3", baudrate=4500000)

    # Wait a moment before starting
    time.sleep(1.0)

    emg_thread = threading.Thread(target=read_EMG, args=(EMG_queue,), daemon=True)
    emg_thread.start()
    time.sleep(1.0)

    close = False

    input()
    i = 0
    last_time = time.time()
    trial_start_time = time.time()

    try:
        while not stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                
                dt = current_time - last_time
                last_time = current_time
                
                try:
                    activation = EMG_queue.get_nowait()
                except queue.Empty:
                    pass

                
                step = (MOTOR_POS_MAX - MOTOR_POS_MIN)/140
                motor_pos = motor.get_position(motor_id=motor.motor_ids[0])
                
                # TODO: This might need to only be one activation since i dont know if there is a antagonist muscle for the finger.
                if activation[0] > 0.2: # Flexion
                    torque = 4.1 * activation[0]  # Scale torque by activation level
                elif activation[1] > 0.2: # Extension
                    torque = -4.1 * activation[1]  # Scale torque by activation level
                else: torque = 0.0
                

                torque_clipped = np.clip(torque, TORQUE_MIN, TORQUE_MAX)
                current = motor.torq2curcom(torque_clipped)

                if motor_pos > MOTOR_POS_MAX and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                    print("Motor at MAX position, stopping positive torque: {}".format(torque_clipped))
                elif motor_pos < MOTOR_POS_MIN and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                    print("Motor at MIN position, stopping negative torque: {}".format(torque_clipped))
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                i += 1
    except Exception as e:
        print(f"Exception during final run: {e}")

    # Stop EMG reading thread and EMG sensor
    motor.sendMotorCommand(motor.motor_ids[0], 0)
    print("Shutting down...")
    stop_event.set()
    motor.close()