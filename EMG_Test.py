from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# General imports
import numpy as np
import queue
import threading
import signal
import time
import matplotlib.pyplot as plt

SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

stop_event = threading.Event()

def read_EMG(raw_queue):
    """EMG读取线程"""
    # Initialize filters
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    emg = DelsysEMG(channel_range=(0,1))
    #emg = DelsysEMG(channel_range=(0,16))
    emg.start()
    while not stop_event.is_set():
        reading = emg.read()

        filtered_bicep = filter_bicep.filter(reading[0])
        filtered_tricep = filter_tricep.filter(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms = filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))
        filtered_tricep_rms = filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))

        activation = interpreter.compute_activation(filtered_bicep_rms, filtered_tricep_rms)
        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])

        try:
            raw_queue.put_nowait((reading, activation, desired_angle_deg))
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait((reading, activation, desired_angle_deg))
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

        # try:
        #     raw_queue.put_nowait(reading)
        # except queue.Full:
        #     try:
        #         raw_queue.get_nowait()
        #         raw_queue.put_nowait(reading)
        #     except queue.Full:
        #         pass
        # except Exception as e:
        #     print(f"[reader] error: {e}", file=sys.stderr)

def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    raw_queue = queue.Queue(maxsize=5)
    reader_thread = threading.Thread(target=read_EMG, args=(raw_queue), daemon=True)
    reader_thread.start()

    raw_bicep_log = []
    raw_tricep_log = []
    activation_log = []
    desired_angle_log = []

    while not stop_event.is_set():
        try:
            reading, activation, desired_angle = raw_queue.get_nowait()
            print(f"Raw EMG: {reading}, Activation: {activation}, Desired Angle: {desired_angle}")
        except queue.Empty:
            continue

        raw_bicep_log.append(reading[0])
        raw_tricep_log.append(reading[1])
        activation_log.append(activation)
        desired_angle_log.append(desired_angle)

        #TODO: uncomment in
        time.sleep(0.005)
    reader_thread.join()

    # plot logs
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(raw_bicep_log, label='Raw Bicep EMG')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Raw Bicep EMG Signal')
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(raw_tricep_log, label='Raw Tricep EMG', color='orange')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Raw Tricep EMG Signal')
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 3)
    activation_array = np.array(activation_log)
    plt.plot(activation_array[:, 0], label='Bicep Activation', color='green')
    plt.plot(activation_array[:, 1], label='Tricep Activation', color='red')
    plt.xlabel('Samples')
    plt.ylabel('Activation Level')
    plt.title('Muscle Activation Levels')
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(desired_angle_log, label='Desired Angle', color='purple')
    plt.xlabel('Samples')
    plt.ylabel('Angle (degrees)')
    plt.title('Desired Joint Angle')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
