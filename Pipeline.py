from EMGSensor import DelsysEMG
from Filtering import rt_filtering
import numpy as np
import time
import matplotlib.pyplot as plt
import queue
from collections import deque
import threading

# windowsize = 50
# data = queue.Queue()
SENTINEL = object()

# Thread function that reads the EMG data and appends to queue
def read_EMG(emg, data):
    while not stop_event.is_set():
        reading = emg.read()
        data.put(reading)

# Thread function that processes the EMG data from queue and appends to lists
def filter_EMG(filter, data, filtered_signal, rms_signal, window_size):
    while not stop_event.is_set():
        if data.qsize() > window_size:
            chunk = [data.get() for _ in range(window_size)]
            filtered, rms = filter.process_chunk(chunk)
            filtered_signal.extend(filtered)
            rms_signal.extend(rms)



if __name__ == "__main__":
    # Initialize variables
    window_size = 50
    data = queue.Queue(maxsize=500)
    filtered_signal = []
    rms_signal = []
    stop_event = threading.Event()

    # Define filter parameters
    sample_rate = 2148  # Hz
    lp_cutoff = 300  # Hz
    hp_cutoff = 20  # Hz
    filter_order = 2

    # Initialize EMG class and filtering class
    filter = rt_filtering(sample_rate, lp_cutoff, hp_cutoff, filter_order)
    emg = DelsysEMG()
    emg.start()

    # Create and start threads
    t_reader = threading.Thread(target=read_EMG, args=(emg, data))
    t_filter = threading.Thread(target=filter_EMG, args=(filter, data, filtered_signal, rms_signal, window_size))

    t_reader.start()
    t_filter.start()

    print("Threads started! To stop press q")

    while True:
        if input().strip().lower() == 'q':
            break
    
    

