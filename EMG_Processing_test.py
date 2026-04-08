# Custom includes
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# TODO: add includes
import numpy as np
import matplotlib.pyplot as plt
import queue
import time

'''
Tests: firstly just EMG no IMU, then test the best performing ones with both EMG and IMU.
'''

# Define global parameters
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

FS = 2000 #Hz
# FS = 1259 #Hz

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

SAVE_PATH = "Outputs/ALLEMG"

if __name__ == "__main__":
    print("Initializing EMG's...")
    # Define filters and interpretors for EMG processing
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=2, order=2)
    # desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=3, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USERNAME, BicepEMG=True, TricepEMG=True)

    # Initialize queues for EMG data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # Initialize EMG sensors
    emg = DelsysEMG(channel_range=(0,1))
    emg.start()
    time.sleep(1)  # Give some time for the EMG to start and stabilize
    
    print("EMG initialized")

    #----------------------------------------------------------------------------------------------------------------------------------
    
    raw_emg_b = []
    raw_emg_t = []
    bandpassed_emg_b = []
    bandpassed_emg_t = []
    rms_emg_b = []
    rms_emg_t = []
    lowpassed_rms_emg_b = []
    lowpassed_rms_emg_t = []
    bicep_a = []
    tricep_a = []
    net_a = []
    lowpassed_net_a = []
    processing_times = []
    print("Press Enter to start test 1: EMG to position no optimization")
    input()
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        t = time.time()
        print(f"elapsed time: {t - start_time:.2f} seconds", end='\r')
        reading = emg.read()

        raw_emg_b.append(reading[0])
        raw_emg_t.append(reading[1])

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        bandpassed_emg_b.append(filtered_bicep)
        bandpassed_emg_t.append(filtered_tricep)

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        rms_emg_b.append(Bicep_RMS)
        rms_emg_t.append(Tricep_RMS)

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        lowpassed_rms_emg_b.append(filtered_bicep_rms)
        lowpassed_rms_emg_t.append(filtered_tricep_rms)

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])

        bicep_a.append(activation[0])
        tricep_a.append(activation[1])
        net_a.append(activation[0] - activation[1])
        lowpassed_net_a.append(float(net_a_lowpass.lowpass(np.atleast_1d(net_a[-1]))[0]))
           
        last_time = time.time()
        processing_times.append(time.time() - t)
    
    print(f"length of test1_desired_angles: {len(raw_emg_b)}, frequency {(len(raw_emg_b)/10):.2f} Hz, average processing time {10/len(raw_emg_b)} ms")
    print(f"mean processing time: {np.mean(processing_times)*1000:.4f} ms, std processing time: {np.std(processing_times)*1000:.2f} ms")

    emg.stop()

    # plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(raw_emg_b)) / FS, raw_emg_b, label='Raw Bicep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Raw Bicep EMG')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(raw_emg_t)) / FS, raw_emg_t, label='Raw Tricep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Raw Tricep EMG')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(bandpassed_emg_b)) / FS, bandpassed_emg_b, label='Bandpassed Bicep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Bandpassed Bicep EMG')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(bandpassed_emg_t)) / FS, bandpassed_emg_t, label='Bandpassed Tricep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Bandpassed Tricep EMG')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(rms_emg_b)) / FS, rms_emg_b, label='RMS Bicep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('RMS Bicep EMG')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(rms_emg_t)) / FS, rms_emg_t, label='RMS Tricep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('RMS Tricep EMG')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(lowpassed_rms_emg_b)) / FS, lowpassed_rms_emg_b, label='Lowpassed RMS Bicep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Lowpassed RMS Bicep EMG')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(lowpassed_rms_emg_t)) / FS, lowpassed_rms_emg_t, label='Lowpassed RMS Tricep EMG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Lowpassed RMS Tricep EMG')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(bicep_a)) / FS, bicep_a, label='Bicep Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.ylim(-0.1, 1.1)
    plt.title('Bicep Activation')
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(len(tricep_a)) / FS, tricep_a, label='Tricep Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.ylim(-0.1, 1.1)
    plt.title('Tricep Activation')
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(len(net_a)) / FS, net_a, label='Net Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.title('Net Activation (Bicep - Tricep)')
    plt.ylim(-1.1, 1.1)
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(lowpassed_net_a)) / FS, lowpassed_net_a, label='Lowpassed Net Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.title('Lowpassed Net Activation')
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()


    