import numpy as np
import matplotlib.pyplot as plt
import queue
import pandas as pd

from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

Sensor_channels = [0, 10] # Bicep, Tricep
User_name = 'VictorBNielsen'

if __name__ == "__main__":
    # Initialize variables
    WindowSize = 50
    Bicep_data_file = []
    Tricep_data_file = []
    Bicep_data = []
    Tricep_data = []
    filtered_Bicep = []
    filtered_Tricep = []
    Bicep_RMS_queue = queue.Queue(maxsize=WindowSize)
    Tricep_RMS_queue = queue.Queue(maxsize=WindowSize)
    Bicep_RMS = []
    Tricep_RMS = []
    filtered_bicep_RMS = []
    filtered_tricep_RMS = []
    positions = []

    # Define filter parameters
    sample_rate = 2000  # Hz # This one is correct according to Trigno Utility control panel
    
    # Variables for plotting
    seconds = 10

    # Initialize filtering class and interpreter class
    filter = rt_filtering(sample_rate, 450, 20, 2)
    # interpreter = PMC(theta_min=0, theta_max=140, user_name=User_name, BicepEMG=True, TricepEMG=True)
    interpreter = PMC(theta_min=0, theta_max=140, user_name=User_name, BicepEMG=True, TricepEMG=False)

    #Load emg data saved in user calibration folder
    df = pd.read_csv(f'Calib/Users/{User_name}/test_raw_data.csv')
    # Remove square brackets and convert to numeric:
    df['Bicep_Raw'] = pd.to_numeric(df['Bicep_Raw'].astype(str).str.replace(r'[\[\]]', '', regex=True), errors='coerce').fillna(0.0)
    df['Tricep_Raw'] = pd.to_numeric(df['Tricep_Raw'].astype(str).str.replace(r'[\[\]]', '', regex=True), errors='coerce').fillna(0.0)
    
    Bicep_data_file = df['Bicep_Raw']
    Tricep_data_file = df['Tricep_Raw']
    print("EMG data loaded!")

    for i in range(len(Bicep_data_file)):
        Bicep_data.append(float(Bicep_data_file[i]))
        Tricep_data.append(float(Tricep_data_file[i]))

        # Filter the Emg signal
        filtered_Bicep.append(filter.bandpass(np.atleast_1d(Bicep_data[i])))
        filtered_Tricep.append(filter.bandpass(np.atleast_1d(Tricep_data[i])))
        
        # Calculate RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_Bicep[-1].item())
        Bicep_RMS.append(np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2)))
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_Tricep[-1].item())
        Tricep_RMS.append(np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2)))

        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_bicep_RMS.append(filter.lowpass(np.atleast_1d(Bicep_RMS[-1])))
        filtered_tricep_RMS.append(filter.lowpass(np.atleast_1d(Tricep_RMS[-1])))

        # Compute activation
        # a_bicep, a_tricep = interpreter.compute_activation([Bicep_RMS[-1].item(), Tricep_RMS[-1].item()])
        a_bicep, a_tricep = interpreter.compute_activation(filtered_bicep_RMS[-1].item())
        angle = interpreter.compute_angle(a_bicep, a_tricep)
        print(f"Angle: {angle}")
        positions.append(angle)
    
    # Plot positions and RMS data
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(filtered_bicep_RMS, label='Bicep RMS', color='blue')
    plt.title('RMS of Bicep EMG Signals')
    plt.xlabel('Samples')
    plt.ylabel('RMS Value')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(filtered_tricep_RMS, label='Tricep RMS', color='red')
    plt.title('RMS of Tricep EMG Signals')
    plt.xlabel('Samples')
    plt.ylabel('RMS Value')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(positions, label='Computed Position', color='green')
    plt.title('Computed Joint Position from EMG Signals')
    plt.xlabel('Samples')
    plt.ylabel('Position (degrees)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    print("Test finished! plotting data...")
    Bicep_data.clear()
    Tricep_data.clear()


