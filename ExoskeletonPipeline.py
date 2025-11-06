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

def send_motor_command(motor, position_queue):
    while not stop_event.is_set():
        try:
            position = position_queue.get(timeout=0.5)
        except queue.Empty:
            # no command available yet; loop again and check stop_event
            continue

        try:
            #motor.sendMotorCommand(motor.motor_ids[0], position)
            i = 0
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)
        finally:
            # rate-limit motor commands
            time.sleep(0.1)

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create EMG sensor instance and setup thread
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    position_queue = queue.Queue(maxsize=SAMPLE_RATE)
    emg = DelsysEMG()

    # Initialize filtering and interpretors
    filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    interpreter.set_Kp(8)
    motor = Motors()

    emg.start()
    # Start EMG reading thread
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_motor = threading.Thread(target=send_motor_command, args=(motor, position_queue), daemon=True)
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
            reading = raw_data.get(timeout=0.5)
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
            Bicep_RMS_queue.put(filtered_Bicep, timeout=0.1)
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
        i += 1
        if i % 100 == 0:
            # periodic status: iteration and queue sizes
            print(f"iteration={i}, raw_q={raw_data.qsize()}, pos_q={position_queue.qsize()}, rms_q={Bicep_RMS_queue.qsize()}")
        print(f"Angle: {position}, iteration: {i}")
        step = 1550/140
        step_offset = 1000
        position = 2550 - int(position*step + step_offset)
        print(f"Position: {position}\n")

        # TODO: Add OIAC and communication with the exoskeleton motor
        #Controller
        #Motor command
        # Put motor commands with a small timeout; if the queue is full
        # discard oldest entry and try again once. Avoid blocking forever.
        if position == last_position:
            # nothing changed; skip
            pass
        else:
            try:
                position_queue.put(position, timeout=0.1)
            except queue.Full:
                try:
                    # discard oldest and try once more
                    position_queue.get_nowait()
                    position_queue.put_nowait(position)
                except Exception as e:
                    print(f"[position queue] could not enqueue: {e}", file=sys.stderr)
        # if not position == last_position:
        #     motor.sendMotorCommand(motor.motor_ids[0], position)
        last_position = position

    # Stop EMG reading thread and EMG sensor
    print("Shutting down...")
    motor.close()
    stop_event.set()
    t_emg.join()
    t_motor.join()
    emg.stop()
    # empty all queues
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    position_queue.queue.clear()
    print("Goodbye!")
