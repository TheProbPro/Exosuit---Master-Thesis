from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_desired_Angle_lowpass, rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# General imports
import numpy as np
import queue
import threading
import signal
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CONTROL_SAMPLE_RATE = 166.7  # Hz
SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140
SAVE_PATH = Path(f"C:/Users/nvigg/Documents/GitHub/Exosuit---Master-Thesis/Outputs/{Path(__file__).stem}/{USER_NAME}/")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


plot_raw_bicep_emg = []
plot_raw_tricep_emg = []
plot_processed_bicep_emg = []
plot_processed_tricep_emg = []
plot_bicep_activation = []
plot_tricep_activation = []
plot_desired_angle = []
plot_control_desired_angle = []

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
    time.sleep(1.0)

    while not stop_event.is_set():
        reading = emg.read()

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        #Plotting
        plot_raw_bicep_emg.append(reading[0])
        plot_raw_tricep_emg.append(reading[1])

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

        #Plotting
        plot_processed_bicep_emg.append(filtered_bicep_rms)
        plot_processed_tricep_emg.append(filtered_tricep_rms)

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])

        #Plotting
        plot_bicep_activation.append(activation[0])
        plot_tricep_activation.append(activation[1])
        plot_desired_angle.append(desired_angle_deg)

        try:
            raw_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    control_loop_frequency = 166.7  # Hz
    dt = 1/control_loop_frequency
    raw_queue = queue.Queue(maxsize=5)
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=CONTROL_SAMPLE_RATE, lp_cutoff=3, order=2)

    print("press enter to start EMG reading thread...")
    input()

    reader_thread = threading.Thread(target=read_EMG, args=(raw_queue,), daemon=True)
    reader_thread.start()
    time.sleep(1.0)


    t = time.time()
    while not stop_event.is_set() and (time.time() - t) < 10:
        print("elapsed time:", time.time() - t, end='\r')
        try:
            desired_angle_deg = desired_angle_lowpass.lowpass(np.atleast_1d(raw_queue.get_nowait()))[0]
        except queue.Empty:
            continue

        plot_control_desired_angle.append(desired_angle_deg)

        time.sleep(dt)

    stop_event.set()
    reader_thread.join()


    # Save data to .csv
    print("Length of recorded arrays: Raw Bicep EMG:", len(plot_raw_bicep_emg), "Raw Tricep EMG:", len(plot_raw_tricep_emg),
          "Processed Bicep EMG:", len(plot_processed_bicep_emg), "Processed Tricep EMG:", len(plot_processed_tricep_emg),
          "Bicep Activation:", len(plot_bicep_activation), "Tricep Activation:", len(plot_tricep_activation),
          "Desired Angle (Thread):", len(plot_desired_angle), "Desired Angle (Control Loop):", len(plot_control_desired_angle))
    df_thread = pd.DataFrame({
        'Raw_Bicep_EMG': plot_raw_bicep_emg,
        'Raw_Tricep_EMG': plot_raw_tricep_emg,
        'Processed_Bicep_EMG': plot_processed_bicep_emg,
        'Processed_Tricep_EMG': plot_processed_tricep_emg,
        'Bicep_Activation': plot_bicep_activation,
        'Tricep_Activation': plot_tricep_activation,
        'Desired_Angle_Thread': plot_desired_angle,
    })
    df_thread.to_csv(SAVE_PATH / f"EMG_Data_{USER_NAME}.csv", index=False)
    df_control = pd.DataFrame({
        'Desired_Angle_Control_Loop': plot_control_desired_angle,
    })
    df_control.to_csv(SAVE_PATH / f"EMG_Control_Loop_Data_{USER_NAME}.csv", index=False)

    # plot emg signal from thread and control loop
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(plot_raw_bicep_emg, label='Raw Bicep EMG (Thread)', alpha=0.5)
    plt.plot(plot_raw_tricep_emg, label='Raw Tricep EMG (Thread)', alpha=0.5)
    plt.xlabel('Samples')
    plt.xlim(0, len(plot_raw_bicep_emg))
    plt.ylabel('EMG Signal (mV)')
    plt.title('Raw EMG Signals')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(plot_processed_bicep_emg, label='Processed Bicep EMG (Thread)', alpha=0.5)
    plt.plot(plot_processed_tricep_emg, label='Processed Tricep EMG (Thread)', alpha=0.5)
    plt.xlabel('Samples')
    plt.xlim(0, len(plot_processed_bicep_emg))
    plt.ylabel('Filtered EMG Signal (mV)')
    plt.title('Processed EMG Signals')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(plot_desired_angle, label='Desired Angle (Thread)', alpha=0.5)
    plt.xlabel('Samples')
    plt.xlim(0, len(plot_desired_angle))
    plt.ylabel('Angle (degrees)')
    plt.title('Desired Angle')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(plot_control_desired_angle, label='Desired Angle (Control Loop)', alpha=0.5, color='orange')
    plt.xlabel('Samples')
    plt.xlim(0, len(plot_control_desired_angle))
    plt.ylabel('Angle (degrees)')
    plt.title('Desired Angle in Control Loop')
    plt.legend()
    plt.tight_layout()
    plt.show()