import numpy as np
import matplotlib.pyplot as plt
import queue
from collections import deque
import threading
import sys
import time

from Filtering import rt_filtering
from EMGSensor import DelsysEMG

# Thread function that reads the EMG data and appends to queue

if __name__ == "__main__":
    # Initialize variables
    window_size = 50
    data = queue.Queue(maxsize=2148)
    
    # Define filter parameters
    sample_rate = 2148  # Hz # This one is correct according to Trigno Utility control panel
    # sample_rate = 2000  # Hz # This one is maybe more correct according to the EMG sensor documentation
    lp_cutoff = 300  # Hz
    hp_cutoff = 20  # Hz
    filter_order = 2

    # Variables for plotting
    seconds = 10
    filtered_signal = []
    rms_signal = []

    # Initialize EMG class and filtering class
    filter = rt_filtering(sample_rate, lp_cutoff, hp_cutoff, filter_order)
    emg = DelsysEMG()
    emg.start()

    

    print("Aquisition started!")
    TIME = time.time()
    while ((time.time() - TIME < seconds)):
        TIME_PROCESS = time.time()
        # Read data
        reading = emg.read()
        data.put(reading)

        if data.qsize() >= window_size:
            chunk = np.asarray([data.get() for _ in range(window_size)]).squeeze().ravel()
            filtered, rms = filter.process_chunk(chunk)
            filtered_signal.extend(filtered)
            rms_signal.extend(rms)

        print("Total acquisition and filtering time: ", time.time() - TIME_PROCESS)

    print("Time elapsed: ", time.time() - TIME, " Expected samples: ", (time.time() - TIME) * sample_rate, " Actual filtered samples samples: ", len(filtered_signal), " Actual rms samples: ", len(rms_signal), " Data in queue: ", data.qsize())
        
    emg.stop()

    # print("data length {}, filtered signal length {}, rms signal length {}".format(len(data.queue),len(filtered_signal), len(rms_signal)))

    # Plot the recorded data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(filtered_signal, label='Filtered EMG', color='blue')
    plt.title('Filtered EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(rms_signal, label='RMS EMG', color='orange')
    plt.title('RMS of EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('RMS Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    


