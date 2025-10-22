import numpy as np
import matplotlib.pyplot as plt
import queue
import sys
import time

from SignalProcessing.Filtering import rt_filtering

if __name__ == "__main__":
    # Load data from .csv file
    #data = np.loadtxt('Outputs/RecordedEMG/EMG_SlowMovement.csv', delimiter=',')
    data = np.loadtxt('Outputs/RecordedEMG/EMG_FastMovement.csv', delimiter=',')

    # Initialize variables
    window_size = 50
    window = queue.Queue(maxsize=window_size)

    # Define filter parameters
    sample_rate = 2000  # Hz # This one is correct according to Trigno Utility control panel
    # sample_rate = 2000  # Hz # This one is maybe more correct according to the EMG sensor documentation
    lp_cutoff = 450  # Hz
    hp_cutoff = 20  # Hz
    filter_order = 4

    # Variables for plotting
    filtered_signal = []
    rms_signal = []
    filtered_RMS = []

    # Initialize EMG class and filtering class
    filter = rt_filtering(sample_rate, lp_cutoff, hp_cutoff, filter_order)

    for reading in data:
        #print(type(reading), reading)
        # Filter data
        #filtered_signal.append(filter.bandpass(np.atleast_1d(reading))*2000) # Scale with gain coefficient from paper
        filtered_signal.append(filter.bandpass(np.atleast_1d(reading))) # Scale with gain coefficient from paper
        # Calculate RMS
        if window.full():
            window.get()
        window.put(filtered_signal[-1].item())
        if not window.full():
            rms_signal.append(0.0)
        else:
            rms_signal.append(np.sqrt(np.mean(np.array(list(window.queue))**2)))
        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_RMS.append(filter.lowpass(np.atleast_1d(rms_signal[-1])))



    print("data length {}, filtered signal length {}, rms signal length {}".format(len(data),len(filtered_signal), len(rms_signal)))

    # Plot the recorded data
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Raw EMG', color='green')
    plt.title('Raw EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.subplot(4, 1, 2)
    plt.plot(filtered_signal, label='Filtered EMG', color='blue')
    plt.title('Filtered EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.plot(rms_signal, label='RMS EMG', color='orange')
    plt.title('RMS of EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('RMS Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(filtered_RMS, label='Filtered RMS EMG', color='red')
    plt.title('Filtered RMS of EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Filtered RMS Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    


