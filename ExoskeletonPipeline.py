# My local imports (EMG sensor, filtering, interpretors, OIAC)
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Motors.DynamixelHardwareInterface import Motors

# General imports
import numpy as np
import queue
import threading
import sys
import signal
import time

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

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
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            # 如果电机支持扭矩控制，使用command[0]
            # 如果只支持位置控制，使用command[1]
            # 这里假设使用位置控制作为后备
            motor.sendMotorCommand(motor.motor_ids[0], command[1])
            motor_state['position'] = motor.get_position()[0]
            motor_state['velocity'] = motor.get_velocity()[0]
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create EMG sensor instance and setup thread
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    position_queue = queue.Queue(maxsize=SAMPLE_RATE)
    motor_state = {'position': 0, 'velocity': 0}
    emg = DelsysEMG()

    # Initialize filtering and interpretors
    filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    interpreter.set_Kp(8)
    motor = Motors()

    # Wait a moment before starting
    time.sleep(1.0)
    motor.sendMotorCommand(motor.motor_ids[0], 2550)
    time.sleep(1.0)

    emg.start()
    # Start EMG reading thread
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_motor = threading.Thread(target=send_motor_command, args=(motor, position_queue, motor_state), daemon=True)
    t_emg.start()
    t_motor.start()
    print("EMG and motor command threads started!")

    # Filter and interpret the raw data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
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
        filtered_Bicep = filter.bandpass(reading[0])

        # Calculate RMS
        try:
            # keep a rolling buffer; if full, drop oldest
            if Bicep_RMS_queue.full():
                try:
                    Bicep_RMS_queue.get_nowait()
                except queue.Empty:
                    pass
            Bicep_RMS_queue.put_nowait(filtered_Bicep)
        except queue.Full:
            # still full after attempt; log and skip this sample
            print("[RMS queue] full, skipping sample", file=sys.stderr)
        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
            
        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_bicep_RMS = filter.lowpass(np.atleast_1d(Bicep_RMS))

        # Compute activation and joint torque
        activation = interpreter.compute_activation(filtered_bicep_RMS)#[0]
        #joint_torque = interpreter.compute_torque(activation)
        #print(f"Joint torque: {int(joint_torque) - 4}")
        position = interpreter.compute_angle(activation[0], activation[1])
        
        step = 1500/140
        step_offset = 1050
        position = 2550 - int(position*step)

        # TODO: Add OIAC and communication with the exoskeleton motor
        #Controller
        #Motor command
        # Put motor commands with a small timeout; if the queue is full
        # discard oldest entry and try again once. Avoid blocking forever.
        try:
            position_queue.put_nowait(position)
        except queue.Full:
            try:
                # discard oldest and try once more
                position_queue.get_nowait()
                position_queue.put_nowait(position)
            except Exception as e:
                print(f"[position queue] could not enqueue: {e}", file=sys.stderr)

    # Stop EMG reading thread and EMG sensor
    print("Shutting down...")
    stop_event.set()
    t_emg.join()
    t_motor.join()
    emg.stop()
    motor.close()
    # empty all queues
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    position_queue.queue.clear()
    print("Goodbye!")
