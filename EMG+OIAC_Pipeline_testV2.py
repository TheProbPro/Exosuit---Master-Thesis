# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Motors.DynamixelHardwareInterface import Motors
from OIAC_Controllers import ada_imp_con

# General imports
import numpy as np
import queue
import threading
import sys
import signal
import time
import matplotlib.pyplot as plt

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

stop_event = threading.Event()

def read_EMG(EMG_sensor, raw_queue):
    """EMG读取线程"""
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            raw_queue.put_nowait(reading)
        except queue.Full:
            try:
                raw_queue.get_nowait()
                raw_queue.put_nowait(reading)
            except queue.Full:
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)

def send_motor_command(motor, command_queue, motor_state):
    """电机命令发送线程"""
    while not stop_event.is_set():
        try:
            # command = (torque, position_fallback)
            command = command_queue.get_nowait()
        except queue.Empty:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
            continue

        try:
            torque = command[0]
            current = motor.torq2curcom(torque)
            #print("motor torque: ", torque, "motor position: ", motor_state['position'])
            if motor_state['position'] < 1050 and torque < 0:
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            elif motor_state['position'] > 2550 and torque > 0:
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            else:
                motor.sendMotorCommand(motor.motor_ids[0], current)
            motor_state['position'] = motor.get_position()[0]
            motor_state['velocity'] = motor.get_velocity()[0]
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)
        
        time.sleep(0.005)  # Sleep briefly to yield CPU

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create EMG sensor instance and setup thread
    raw_data = queue.Queue(maxsize=5)
    desired_pos_smoothing_queue = queue.Queue(maxsize=5)
    plot_position = []
    plot_desired_position = []
    plot_error = []
    plot_torque = []

    emg = DelsysEMG(channel_range=(0, 1))

    # Initialize filtering and interpretors
    bicep_filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    tricep_filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    pos_filter = rt_filtering(SAMPLE_RATE, 10, 2, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    
    motor = Motors()

    # Wait a moment before starting
    time.sleep(1.0)

    emg.start()
    # Start EMG reading thread
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_emg.start()
    print("EMG and motor command threads started!")

    # Filter and interpret the raw data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    joint_torque = 0.0
    last_desired_angle = 0
    i = 0
    OIAC = ada_imp_con(1) # 1 degree of freedom

    last_time = time.time()
    loop_timer = time.time()
    #while not stop_event.is_set():
    while last_time - loop_timer < 10:  # Run for 10 seconds
        # Use a blocking get with timeout to avoid busy-waiting and to
        # allow the reader thread to drive the queue at its own rate.
        try:
            reading = raw_data.get_nowait()
        except queue.Empty:
            # no new raw data; yield CPU briefly and continue
            time.sleep(0.001)
            continue
        
        current_time = time.time()
        dt = current_time - last_time


        # Filter data
        filtered_Bicep = bicep_filter.bandpass(np.atleast_1d(reading[0]))
        filtered_Tricep = tricep_filter.bandpass(np.atleast_1d(reading[1]))
        print(f"filtered_Bicep: {filtered_Bicep}, filtered_Tricep: {filtered_Tricep}")

        # Calculate RMS
        if Bicep_RMS_queue.full():
            try:
                Bicep_RMS_queue.get_nowait()
            except queue.Empty:
                pass
        Bicep_RMS_queue.put_nowait(filtered_Bicep)

        if Tricep_RMS_queue.full():
            try:
                Tricep_RMS_queue.get_nowait()
            except queue.Empty:
                pass
        Tricep_RMS_queue.put_nowait(filtered_Tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))
            
        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_bicep_RMS = bicep_filter.lowpass(np.atleast_1d(Bicep_RMS))
        filtered_tricep_RMS = tricep_filter.lowpass(np.atleast_1d(Tricep_RMS))
        print(f"filtered_bicep_RMS: {filtered_bicep_RMS}, filtered_tricep_RMS: {filtered_tricep_RMS}")

        # Compute activation and joint torque
        activation = interpreter.compute_activation([filtered_bicep_RMS, filtered_tricep_RMS])
        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
        print(f"activation: {activation}, desired_angle_deg: {desired_angle_deg}")
        
        #TODO: Test with and without smoothing of position
        if desired_pos_smoothing_queue.full():
            desired_pos_smoothing_queue.get_nowait()
        desired_pos_smoothing_queue.put_nowait(float(desired_angle_deg))
        numpyarray = np.array(list(desired_pos_smoothing_queue.queue))
        average_position = np.mean(numpyarray)
        # Alternative to windowed average
        #average_position = pos_filter.lowpass(np.atleast_1d(desired_angle_deg))
        plot_desired_position.append(average_position)

        desired_angle_rad = math.radians(average_position)
        desired_velocity = (average_position - last_desired_angle) / dt if dt > 0 else 0.0
        last_desired_angle = average_position
        step = 1500/140
        current_velocity = motor.get_velocity()
        motor_pos = motor.get_position()
        current_angle_deg = (2550 - motor_pos) / step
        current_angle_rad = math.radians(current_angle_deg)
        plot_position.append(current_angle_deg)
        plot_error.append(current_angle_deg - average_position)

        # print(f"current_angle_deg: {current_angle_deg}, desired_angle_deg: {average_position}, error: {current_angle_deg - average_position}, current_velocity: {current_velocity}, desired_velocity: {desired_velocity}")

        # OIAC online impedance adaptation
        K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity)
        tau_fb = OIAC.calc_tau_fb()[0,0] # TODO: This might have to swap sign
        #print(f"tau_fb: {tau_fb}")
        tau_clipped = np.clip(tau_fb, TORQUE_MIN, TORQUE_MAX)
        plot_torque.append(float(tau_clipped))
        
        position_motor = 2550 - int(desired_angle_deg*step)

        current = motor.torq2curcom(tau_clipped)
        #print("motor torque: ", tau_clipped, "motor position: ", motor_pos)
        
        if motor_pos < 1050 and tau_clipped < 0:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
        elif motor_pos > 2550 and tau_clipped > 0:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
        else:
            motor.sendMotorCommand(motor.motor_ids[0], current)

        time.sleep(0.005)  # Sleep briefly to yield CPU
        
        last_time = current_time

    # Stop EMG reading thread and EMG sensor
    motor.sendMotorCommand(motor.motor_ids[0], 0)
    print("Shutting down...")
    stop_event.set()
    t_emg.join()
    emg.stop()
    motor.close()
    # empty all queues
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    desired_pos_smoothing_queue.queue.clear()
    
    print("plotting data...")
    #plot desired and actual position in one graph and error in another graph
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(plot_position, label='Actual Position (deg)')
    plt.plot(plot_desired_position, label='Desired Position (deg)', linestyle='--')
    plt.title('Actual vs Desired Position')
    plt.xlabel('Time Steps')
    plt.ylabel('Position (deg)')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(plot_error, label='Position Error (deg)', color='red')
    plt.title('Position Error Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Error (deg)')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(plot_torque, label='Applied Torque (Nm)', color='green')
    plt.title('Applied Torque Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Goodbye!")
