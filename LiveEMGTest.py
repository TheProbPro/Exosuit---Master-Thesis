# My local imports (EMG sensor, filtering, interpretors, OIAC)
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# General imports
import numpy as np
import matplotlib.pyplot as plt
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

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # arrays to visualize data
    raw_bicep = []
    raw_tricep = []
    Bicep_processed = []
    Tricep_processed = []
    activation_bicep = []
    activation_tricep = []
    position_list = []

    # Create EMG sensor instance and setup thread
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    position_queue = queue.Queue(maxsize=SAMPLE_RATE)
    emg = DelsysEMG(channel_range=(0,1))

    # Initialize filtering and interpretors
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    interpreter.set_Kp(8)

    emg.start()
    # Start EMG reading thread
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_emg.start()
    print("EMG thread started!")

    # Filter and interpret the raw data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    joint_torque = 0.0
    last_position = 0
    i = 0
    Time = time.time()
    while not stop_event.is_set() and time.time() - Time < 10:  # Run for 60 seconds
        # Use a blocking get with timeout to avoid busy-waiting and to
        # allow the reader thread to drive the queue at its own rate.
        try:
            reading = raw_data.get_nowait()
        except queue.Empty:
            # no new raw data; yield CPU briefly and continue
            time.sleep(0.001)
            continue

        # For plotting
        raw_bicep.append(reading[0])
        raw_tricep.append(reading[1])

        # Filter data
        filtered_Bicep = filter_bicep.bandpass(reading[0])
        filtered_Tricep = filter_tricep.bandpass(reading[1])

        # Calculate RMS
        try:
            # keep a rolling buffer; if full, drop oldest
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
        except queue.Full:
            # still full after attempt; log and skip this sample
            print("[RMS queue] full, skipping sample", file=sys.stderr)
        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))
        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_bicep_RMS = filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))
        filtered_tricep_RMS = filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))

        # for plotting
        Bicep_processed.append(filtered_bicep_RMS[0])
        Tricep_processed.append(filtered_tricep_RMS[0])

        # Compute activation and joint torque
        activation = interpreter.compute_activation([filtered_bicep_RMS[0], filtered_tricep_RMS[0]])

        # For plotting
        activation_bicep.append(activation[0])
        activation_tricep.append(activation[1])

        #joint_torque = interpreter.compute_torque(activation)
        #print(f"Joint torque: {int(joint_torque) - 4}")
        position = interpreter.compute_angle(activation[0], activation[1])
        print(f"Computed position: {position} degrees, from activations Bicep: {activation[0]}, Tricep: {activation[1]}")

        #for plotting
        position_list.append(position)
        
        # step = 1500/140
        # step_offset = 1050
        # position = 2550 - int(position*step)

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
    emg.stop()
    # empty all queues
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    position_queue.queue.clear()
    print("Goodbye!")

    #Plotting
    plt.figure(figsize=(10, 6))
    # Raw Bicep and tricep
    plt.subplot(4, 1, 1)
    plt.plot(raw_bicep, label='Raw Bicep EMG', color='red')
    plt.plot(raw_tricep, label='Raw Tricep EMG', color='blue')
    plt.title('Raw EMG Signals')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    # Processed Bicep and tricep
    plt.subplot(4, 1, 2)
    plt.plot(Bicep_processed, label='Processed Bicep EMG', color='red')
    plt.plot(Tricep_processed, label='Processed Tricep EMG', color='blue')
    plt.title('Processed EMG Signals')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    # Activation Bicep and tricep
    plt.subplot(4, 1, 3)
    plt.plot(activation_bicep, label='Bicep Activation', color='red')
    plt.plot(activation_tricep, label='Tricep Activation', color='blue')
    plt.title('Muscle Activations')
    plt.xlabel('Samples')
    plt.ylabel('Activation Level')
    plt.legend()
    plt.grid()
    # Position
    plt.subplot(4, 1, 4)
    plt.plot(position_list, label='Computed Position', color='green')
    plt.title('Computed Joint Position')
    plt.xlabel('Samples')
    plt.ylabel('Position (degrees)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()