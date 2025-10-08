import numpy as np
import matplotlib.pyplot as plt
import queue
from collections import deque
import threading
import sys
import time

from SignalProcessing.Filtering import rt_filtering
from Sensors.EMGSensor import DelsysEMG

# Thread function that reads the EMG data and appends to queue
# IMPORTANT: DO NOT READ FROM THREAD. We lose around 14000 samples over 10 seconds if we do that.
def read_EMG(emg, data, stop_event):
    while not stop_event.is_set():
        TIME = time.time()
        reading = emg.read()
        data.put(reading)
        print("Read time: ", time.time() - TIME)

# Thread function that processes the EMG data from queue and appends to lists
def filter_EMG(filter, data, filtered_signal, rms_signal, window_size, stop_event):
    while not stop_event.is_set():
        TIME = time.time()
        if data.qsize() >= window_size:
            chunk = np.asarray([data.get() for _ in range(window_size)]).squeeze().ravel()
            filtered, rms = filter.process_chunk(chunk)
            filtered_signal.extend(filtered)
            rms_signal.extend(rms)
        print("Filter time: ", time.time() - TIME)

# def wait_for_q(stop_event):
#     for line in sys.stdin:
#         if line.strip().lower() == 'q':
#             stop_event.set()
#             break

if __name__ == "__main__":
    # Initialize variables
    window_size = 50
    data = queue.Queue(maxsize=2148)
    stop_event = threading.Event()

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

    print("EMG started!")

    # Create and start threads
    #t_reader = threading.Thread(target=read_EMG, args=(emg, data, stop_event))
    t_filter = threading.Thread(target=filter_EMG, args=(filter, data, filtered_signal, rms_signal, window_size, stop_event))
    # t_listener = threading.Thread(target=wait_for_q, args=(stop_event,))

    #t_reader.start()
    t_filter.start()
    # t_listener.start()

    print("Threads started! To stop press q")
    TIME = time.time()
    while ((time.time() - TIME < seconds)):
        reading = emg.read()
        data.put(reading)
        
    print("Time elapsed: ", time.time() - TIME)
        
    stop_event.set()
    #t_reader.join()
    t_filter.join()
    emg.stop()

    print("data length {}, filtered signal length {}, rms signal length {}".format(len(data.queue),len(filtered_signal), len(rms_signal)))

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
    


