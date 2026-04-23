import numpy as np
import matplotlib.pyplot as plt
import queue
import time
import pandas as pd
import os
import threading

from SignalProcessing.Filtering import rt_filtering
from Sensors.EMGSensor import DelsysEMG

# TODO: Clean this script and optimize it for when tests needs to be performed

Sensor_channels = [0, 1] # Bicep, Tricep
User_name = 'VictorBNielsen'
# User_name = 'ZicehnWang'
# User_name = 'XiaofengXiong'
# User_name = 'Cao'

FS = 2000 #Hz
# FS = 1259 #Hz

stop_event = threading.Event()

def EMG(emg_queue):
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    while not stop_event.is_set():
        reading = emg.read()
        
        # Bandpass filter
        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        #RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        #Low pass filter
        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        try:
            emg_queue.put_nowait((filtered_bicep_rms, filtered_tricep_rms))
        except queue.Full:
            emg_queue.get_nowait()
            emg_queue.put_nowait((filtered_bicep_rms, filtered_tricep_rms))

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

def _calc_MVC(signal, sampling_rate=FS, win_ms=200):
        i = int(np.max(signal))
        w = int(win_ms / 1000 * sampling_rate) // 2
        return np.mean(signal[max(0,i-w): min(i+w+1,len(signal))])

if __name__ == "__main__":
    # Create emg_queue
    emg_queue = queue.Queue(maxsize=3)
    
    # Variables for plotting
    seconds = 10
    sample_rate = FS  # Hz # This one is correct according to Trigno Utility control panel

    # Start thread for EMG data acquisition and processing
    emg_thread = threading.Thread(target=EMG, args=(emg_queue,))
    emg_thread.start()
    time.sleep(5)

    print("EMG started!")

    print("Starting test of data aquisition and filtering for {} seconds...".format(2))
    TIME = time.time()
    bicep = []
    tricep = []
    while ((time.time() - TIME < 2)):
        try:
            filtered_bicep_rms, filtered_tricep_rms = emg_queue.get_nowait()
        except queue.Empty:
            continue
        bicep.append(filtered_bicep_rms)
        tricep.append(filtered_tricep_rms)
    
    print("Test finished! plotting data...")

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(bicep)) / sample_rate, bicep)
    plt.title("Filtered Bicep RMS")
    plt.subplot(2,1,2)
    plt.plot(np.arange(len(tricep)) / sample_rate, tricep)
    plt.title("Filtered Tricep RMS")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.tight_layout()
    plt.show()
    
    # clear all data
    bicep.clear()
    tricep.clear()

    input("Press Enter to start calibration of MVC. After pressing enter rest your arm for the next 10 seconds...")
    TIME = time.time()
    while (time.time() - TIME < 10):
        try:
            filtered_bicep_rms, filtered_tricep_rms = emg_queue.get_nowait()
        except queue.Empty:
            continue
        bicep.append(filtered_bicep_rms)
        tricep.append(filtered_tricep_rms)

    mean_rest_bicep = np.mean(np.array(bicep))
    mean_rest_tricep = np.mean(np.array(tricep))
    print("Rest calibration done, mean rest bicep: {}, mean rest tricep: {}".format(mean_rest_bicep, mean_rest_tricep))
    bicep.clear()
    tricep.clear()


    MVC_trials = 3
    max_bicep = []
    max_tricep = []
    input("Press Enter to start Maximum Voluntary Contraction (MVC) for Proportional Myoelectric Control...")
    for trial in range(MVC_trials):
        input("Press Enter to start trial {} of {}. Then perform MAXIMUM contraction of Bicep for 5 seconds...".format(trial+1, MVC_trials))
        print("Starting trial {}...".format(trial+1))
        TIME = time.time()
        while (time.time() - TIME < 5):
            try:
                filtered_bicep_rms, filtered_tricep_rms = emg_queue.get_nowait()
            except queue.Empty:
                continue
            bicep.append(filtered_bicep_rms)
        input("Press enter to start trial {} of {}. Then perform MAXIMUM contraction of Tricep for 5 seconds...".format(trial+1, MVC_trials))
        print("Starting trial {}...".format(trial+1))
        TIME = time.time()
        while (time.time() - TIME < 5):
            try:
                filtered_bicep_rms, filtered_tricep_rms = emg_queue.get_nowait()
            except queue.Empty:
                continue
            tricep.append(filtered_tricep_rms)
        max_bicep.append(_calc_MVC(bicep, sampling_rate=sample_rate, win_ms=200))
        max_tricep.append(_calc_MVC(tricep, sampling_rate=sample_rate, win_ms=200))
        print("Trial {} done! Max bicep: {}, Max tricep: {}".format(trial+1, max_bicep[-1], max_tricep[-1]))
        
        bicep.clear()
        tricep.clear()

    stop_event.set()
    emg_thread.join()

    # calculate average max
    avg_max_bicep = np.mean(np.array(max_bicep))
    avg_max_tricep = np.mean(np.array(max_tricep))
    print("MVC calibration done, average max bicep: {}, average max tricep: {}".format(avg_max_bicep, avg_max_tricep))
    max_bicep.clear()
    max_tricep.clear()

    # save the calibration data to user csv files
    print("Saving calibration data to Calib/Users/{}/".format(User_name))
    #Check if directory exists, if not create it
    if not os.path.exists("Calib/Users/{}".format(User_name)):
        os.makedirs("Calib/Users/{}".format(User_name))

    df_rest = pd.DataFrame()
    df_rest['Bicep'] = [mean_rest_bicep]
    df_rest['Tricep'] = [mean_rest_tricep]
    df_rest.to_csv(f'Calib/Users/{User_name}/rest_signal.csv', index=False)

    df_max = pd.DataFrame()
    df_max['Bicep'] = [avg_max_bicep]
    df_max['Tricep'] = [avg_max_tricep]
    df_max.to_csv(f'Calib/Users/{User_name}/max_signal.csv', index=False)
    print("Calibration data saved to Calib/Users/{}/".format(User_name))
    
