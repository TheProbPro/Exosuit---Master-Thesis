import numpy as np
import matplotlib.pyplot as plt
import queue
import time
import pandas as pd
import os

from SignalProcessing.Filtering import rt_filtering
from Sensors.EMGSensor import DelsysEMG

# TODO: Clean this script and optimize it for when tests needs to be performed

Sensor_channels = [0, 1] # Bicep, Tricep
User_name = 'VictorBNielsen'
# User_name = 'XiaofengXiong'

def _calc_MVC(signal, sampling_rate=2000, win_ms=200):
        i = int(np.max(signal))
        w = int(win_ms / 1000 * sampling_rate) // 2
        return np.mean(signal[max(0,i-w): min(i+w+1,len(signal))])

if __name__ == "__main__":
    # Initialize variables
    Bicep_data = []
    Tricep_data = []
    filtered_Bicep = []
    filtered_Tricep = []
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    Bicep_RMS = []
    Tricep_RMS = []
    filtered_Bicep_RMS = []
    filtered_Tricep_RMS = []

    # Define filter parameters
    sample_rate = 2000  # Hz # This one is correct according to Trigno Utility control panel
    
    # Variables for plotting
    seconds = 10

    # Initialize EMG class and filtering class
    filter_bicep = rt_filtering(sample_rate, 450, 20, 2)
    filter_tricep = rt_filtering(sample_rate, 450, 20, 2)
    emg = DelsysEMG(channel_range=(0,16))
    emg.start()

    print("EMG started!")

    print("Starting test of data aquisition and filtering for {} seconds...".format(2))
    TIME = time.time()
    while ((time.time() - TIME < 2)):
        # Aquire data
        reading = emg.read()
        Bicep_data.append(reading[Sensor_channels[0]])
        Tricep_data.append(reading[Sensor_channels[1]])

        # Filter data
        filtered_Bicep.append(filter_bicep.bandpass(reading[Sensor_channels[0]]))
        filtered_Tricep.append(filter_tricep.bandpass(reading[Sensor_channels[1]]))

        # Calculate RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_Bicep[-1].item())
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_Tricep[-1].item())
        Bicep_RMS.append(filter_bicep.RMS(list(Bicep_RMS_queue.queue)))
        Tricep_RMS.append(filter_tricep.RMS(list(Tricep_RMS_queue.queue)))

        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_Bicep_RMS.append(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS[-1])))
        filtered_Tricep_RMS.append(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS[-1])))

    emg.stop()
    
    print("Test finished! plotting data...")

    # # Save RMS data as .csv file
    # df_rms = pd.DataFrame()
    # df_rms['Bicep_RMS'] = filtered_Bicep_RMS
    # df_rms['Tricep_RMS'] = filtered_Tricep_RMS
    # df_rms.to_csv(f'Calib/Users/{User_name}/test_rms_data.csv', index=False)

    # df_filtered = pd.DataFrame()
    # df_filtered['Bicep_Filtered'] = filtered_Bicep
    # df_filtered['Tricep_Filtered'] = filtered_Tricep
    # df_filtered.to_csv(f'Calib/Users/{User_name}/test_filtered_data.csv', index=False)

    # df_raw = pd.DataFrame()
    # df_raw['Bicep_Raw'] = Bicep_data
    # df_raw['Tricep_Raw'] = Tricep_data
    # df_raw.to_csv(f'Calib/Users/{User_name}/test_raw_data.csv', index=False)
    # quit()

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(Bicep_data, label='Raw Bicep')
    plt.plot(Tricep_data, label='Raw Tricep')
    plt.title('Raw EMG Data')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(filtered_Bicep, label='Filtered Bicep')
    plt.plot(filtered_Tricep, label='Filtered Tricep')
    plt.title('Filtered EMG Data')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(Bicep_RMS, label='Bicep RMS')
    plt.plot(Tricep_RMS, label='Tricep RMS')
    plt.title('RMS of EMG Data')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(filtered_Bicep_RMS, label='Filtered Bicep RMS')
    plt.plot(filtered_Tricep_RMS, label='Filtered Tricep RMS')
    plt.title('Filtered RMS of EMG Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Calib/Users/{}/test_emg_plot.png'.format(User_name))
    plt.show()
    
    # clear all data
    Bicep_data.clear()
    Tricep_data.clear()
    filtered_Bicep.clear()
    filtered_Tricep.clear()
    Bicep_RMS.clear()
    Tricep_RMS.clear()
    filtered_Bicep_RMS.clear()
    filtered_Tricep_RMS.clear()

    while not Bicep_RMS_queue.empty():
        Bicep_RMS_queue.get()
    while not Tricep_RMS_queue.empty():
        Tricep_RMS_queue.get()

    input("Press Enter to start calibration of MVC. After pressing enter rest your arm for the next 10 seconds...")
    emg.start()
    TIME = time.time()
    while (time.time() - TIME < 10):
        reading = emg.read()
        Bicep_data.append(reading[Sensor_channels[0]])
        Tricep_data.append(reading[Sensor_channels[1]])

        filtered_Bicep.append(filter_bicep.bandpass(reading[Sensor_channels[0]]))
        filtered_Tricep.append(filter_tricep.bandpass(reading[Sensor_channels[1]]))
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_Bicep[-1].item())
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_Tricep[-1].item())
        Bicep_RMS.append(filter_bicep.RMS(list(Bicep_RMS_queue.queue)))
        Tricep_RMS.append(filter_tricep.RMS(list(Tricep_RMS_queue.queue)))

        filtered_Bicep_RMS.append(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS[-1])))
        filtered_Tricep_RMS.append(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS[-1])))

    emg.stop()

    mean_rest_bicep = np.mean(np.array(filtered_Bicep_RMS))
    mean_rest_tricep = np.mean(np.array(filtered_Tricep_RMS))
    print("Rest calibration done, mean rest bicep: {}, mean rest tricep: {}".format(mean_rest_bicep, mean_rest_tricep))
    Bicep_data.clear()
    Tricep_data.clear()
    filtered_Bicep.clear()
    filtered_Tricep.clear()
    Bicep_RMS.clear()
    Tricep_RMS.clear()
    filtered_Bicep_RMS.clear()
    filtered_Tricep_RMS.clear()
    while not Bicep_RMS_queue.empty():
        Bicep_RMS_queue.get()
    while not Tricep_RMS_queue.empty():
        Tricep_RMS_queue.get()


    MVC_trials = 3
    max_bicep = []
    max_tricep = []
    input("Press Enter to start Maximum Voluntary Contraction (MVC) for Proportional Myoelectric Control...")
    for trial in range(MVC_trials):
        input("Press Enter to start trial {} of {}. Then perform MAXIMUM contraction of Bicep for 5 seconds...".format(trial+1, MVC_trials))
        print("Starting trial {}...".format(trial+1))
        emg.start()
        TIME = time.time()
        while (time.time() - TIME < 5):
            reading = emg.read()
            Bicep_data.append(reading[Sensor_channels[0]])

            filtered_Bicep.append(filter_bicep.bandpass(reading[Sensor_channels[0]]))
            if Bicep_RMS_queue.full():
                Bicep_RMS_queue.get()
            Bicep_RMS_queue.put(filtered_Bicep[-1].item())
            Bicep_RMS.append(filter_bicep.RMS(list(Bicep_RMS_queue.queue)))
            filtered_Bicep_RMS.append(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS[-1])))
        emg.stop()
        input("Press enter to start trial {} of {}. Then perform MAXIMUM contraction of Tricep for 5 seconds...".format(trial+1, MVC_trials))
        print("Starting trial {}...".format(trial+1))
        emg.start()
        TIME = time.time()
        while (time.time() - TIME < 5):
            reading = emg.read()
            Tricep_data.append(reading[Sensor_channels[1]])

            filtered_Tricep.append(filter_tricep.bandpass(reading[Sensor_channels[1]]))
            if Tricep_RMS_queue.full():
                Tricep_RMS_queue.get()
            Tricep_RMS_queue.put(filtered_Tricep[-1].item())
            Tricep_RMS.append(filter_tricep.RMS(list(Tricep_RMS_queue.queue)))
            filtered_Tricep_RMS.append(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS[-1])))
        emg.stop()
        max_bicep.append(_calc_MVC(filtered_Bicep_RMS, sampling_rate=sample_rate, win_ms=200))
        max_tricep.append(_calc_MVC(filtered_Tricep_RMS, sampling_rate=sample_rate, win_ms=200))
        print("Trial {} done! Max bicep: {}, Max tricep: {}".format(trial+1, max_bicep[-1], max_tricep[-1]))
        
        Bicep_data.clear()
        Tricep_data.clear()
        filtered_Bicep.clear()
        filtered_Tricep.clear()
        Bicep_RMS.clear()
        Tricep_RMS.clear()
        filtered_Bicep_RMS.clear()
        filtered_Tricep_RMS.clear()
        while not Bicep_RMS_queue.empty():
            Bicep_RMS_queue.get()
        while not Tricep_RMS_queue.empty():
            Tricep_RMS_queue.get()
    emg.stop()

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
    
