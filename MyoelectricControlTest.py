import numpy as np
import matplotlib.pyplot as plt
import queue
from collections import deque
import sys
import time
import pandas as pd
import os

from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

Sensor_channels = [0, 10] # Bicep, Tricep
User_name = 'VictorBNielsen'

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

    # Define filter parameters
    sample_rate = 2000  # Hz # This one is correct according to Trigno Utility control panel
    
    # Variables for plotting
    seconds = 10

    # Initialize filtering class and interpreter class
    filter = rt_filtering(sample_rate, 300, 20, 2)
    interpreter = PMC(theta_min=0, theta_max=140, user_name=User_name, BicepEMG=True, TricepEMG=True)

    #Load emg data saved in user calibration folder


    
