import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfilt_zi, tf2sos
import time

### This class implements various filtering techniques to clean the EMG signal. This is a temporary class that will be optimized to use actual equations later. ###
## The given variables fs, nyquist, Wn, and cutoff are all given and calculated using muscle band ranges. ##

class filtering:
    def __init__(self, sample_rate):
        self.fs = sample_rate  # Sample rate in Hz
        self.nyquist = 0.5 * self.fs # Nyquist frequency/ folding frequency (fN)

    def ButterworthLowPass(self, data, cutoff=450, order=4):
        # Normalized cutoff frequency
        Wn = cutoff / self.nyquist
        # Get the filter coefficients
        b, a = butter(order, Wn, btype='lowpass')
        # Apply the filter
        filtered_data = filtfilt(b, a, data)
    
        return filtered_data
    
    def ButterworthHighPass(self, data, cutoff=20, order=4):
        # Normalized cutoff frequency
        Wn = cutoff / self.nyquist
        # Get the filter coefficients
        b, a = butter(order, Wn, btype='highpass')
        # Apply the filter
        filtered_data = filtfilt(b, a, data)
    
        return filtered_data
    
    def NotchFilter(self, data, notch_freq=50):
        bw = 1.0
        Q = notch_freq / bw
        b, a = iirnotch(notch_freq / (self.fs / 2), Q)
        filtered_data = filtfilt(b, a, data)
    
        return filtered_data
    
    def ButterworthBandPassFilter(self, data, lowcut=20, highcut=450, order=4):
        # Normalized cutoff frequencies
        Wn = [lowcut / self.nyquist, highcut / self.nyquist]
        # Get the filter coefficients
        b, a = butter(order, Wn, btype='bandpass')
        # Apply the filter
        filtered_data = filtfilt(b, a, data)
    
        return filtered_data
    
    def ButterworthBandStopFilter(self, data, lowcut=48, highcut=52, order=4):
        # Normalized cutoff frequencies
        Wn = [lowcut / self.nyquist, highcut / self.nyquist]
        # Get the filter coefficients
        b, a = butter(order, Wn, btype='bandstop')
        # Apply the filter
        filtered_data = filtfilt(b, a, data)
    
        return filtered_data
    
    def PlotFilter(self, data, filtered_data, title="Filter Result", savepath="./AttemptedFiltering/"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Original signal
        axes[0].plot(data, label='Original Data', alpha=0.7)
        axes[0].set_title("Original Signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].grid(True)

        # Filtered signal
        axes[1].plot(filtered_data, label='Filtered Data', linewidth=2, color='orange')
        axes[1].set_title("Filtered Signal")
        axes[1].set_xlabel("Sample Index")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        axes[1].grid(True)

        # Adjust layout and save
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        plt.savefig(savepath + title + ".png")
        plt.show()

    def FFT(self, data, plot=False):
        # Compute the FFT of the signal
        N = len(data)
        Xf = np.fft.fft(data)
        freqs = np.fft.fftfreq(N, d=1/self.fs)

        # Create magniude spectrum
        magnitude = np.abs(Xf) / N
        
        # --- Plot ---
        if plot:
            plt.figure(figsize=(8,4))
            plt.plot(freqs, magnitude)
            #plt.xlim(0, self.fs/2)                   # Nyquist limit
            plt.xlim(0, 200)  # Limit x-axis to 500 Hz for better visibility
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("FFT Magnitude Spectrum of sEMG signal")
            plt.grid(True)
            plt.savefig("AttemptedFiltering/FFT_Original_Signal.png")
            plt.show()

        return freqs, magnitude

    def Windowed_RMS(self, filtered_data, window_size=100):
        rms_values = []
        for i in range(0, len(filtered_data), window_size):
            window = filtered_data[i:i+window_size]
            rms = np.sqrt(np.mean(np.square(np.abs(window))))
            rms_values.append(rms)
        return rms_values

# if __name__ == "__main__":
#     # Sample rate
#     sample_rate = 2148  # Hz
#     # Cutoff Fequencies
#     #low_cutoff = 10  # Hz
#     #high_cutoff = 20  # Hz
#     #high_cutoff = 40 #20
#     high_cutoff = 20 #40
#     low_cutoff = 300 #300
#     #low_cutoff = 50 #300
#     notch_freq = 45  # Hz
#     order = 2  # Order of the filter

#     # Create an instance of the filtering class
#     filter_instance = filtering(sample_rate)

#     # define read and save paths
#     readpath = "./RecordedEMG/"
#     savepath = "./AttemptedFiltering/"
#     # Read recorded EMG data from a CSV file
#     # data = np.loadtxt(readpath + "EMG_Rest.csv", delimiter=",")
#     # data = np.loadtxt(readpath + "EMG_SlowMovement.csv", delimiter=",")
#     data = np.loadtxt(readpath + "EMG_FastMovement.csv", delimiter=",")

#     # Calculate FFT of original signal
#     freqs, magnitude = filter_instance.FFT(data, plot=True)

#     # # Apply filters, plot results and check processing time
#     # # High-pass filter
#     # start_time = time.time()
#     # high_passed_data = filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order)
#     # print(f"High-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, high_passed_data, title="High-Pass_Filter_Result", savepath=savepath)

#     # # Low-pass filter
#     # start_time = time.time()
#     # low_passed_data = filter_instance.ButterworthLowPass(data, cutoff=low_cutoff, order=order)
#     # print(f"Low-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, low_passed_data, title="Low-Pass_Filter_Result", savepath=savepath)

#     # # Notch filter
#     # start_time = time.time()
#     # notch_filtered_data = filter_instance.NotchFilter(data, notch_freq=notch_freq)
#     # print(f"Notch filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, notch_filtered_data, title="Notch_Filter_Result", savepath=savepath)
    
#     # # Combined high-pass and low-pass filter
#     # start_time = time.time()
#     # combined_filtered_data = filter_instance.ButterworthLowPass(
#     #     filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order), cutoff=low_cutoff, order=order)
#     # print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, combined_filtered_data, title="Highpass_Lowpass_Result", savepath=savepath)

#     # # Combined low-pass and high-pass filter
#     # start_time = time.time()
#     # combined_filtered_data_2 = filter_instance.ButterworthHighPass(
#     #     filter_instance.ButterworthLowPass(data, cutoff=low_cutoff, order=order), cutoff=high_cutoff, order=order)
#     # print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, combined_filtered_data_2, title="Lowpass_Highpass_Result", savepath=savepath)

#     # # Combined High-pass, Notch, and low-pass filter
#     # start_time = time.time()
#     # combined_filtered_data_3 = filter_instance.ButterworthLowPass(
#     #     filter_instance.NotchFilter(
#     #         filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order), notch_freq=notch_freq), cutoff=low_cutoff, order=order)
#     # print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, combined_filtered_data_3, title="Highpass_Notch_Lowpass_Result", savepath=savepath)

#     # # band-pass filter
#     # start_time = time.time()
#     # band_passed_data = filter_instance.ButterworthBandPassFilter(data, lowcut=high_cutoff, highcut=low_cutoff, order=order)
#     # print(f"Band-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, band_passed_data, title="Band-Pass_Filter_Result", savepath=savepath)

#     # # RMS on band-passed data
#     # rms_value = filter_instance.Windowed_RMS(band_passed_data, window_size=100)
#     # #plot RMS signal
#     # plt.figure(figsize=(12, 4))
#     # plt.plot(rms_value, label='RMS of Band-Passed Signal', color='green')
#     # plt.title("Windowed RMS of Band-Passed Signal")
#     # plt.xlabel("Window Index")
#     # plt.ylabel("RMS Value")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.savefig(savepath + "RMS_Band-Pass_Signal.png")
#     # plt.show()

#     # Attempt at full preprossessing pipeline
#     # bandpass
#     start_time = time.time()
#     band_passed_data = filter_instance.ButterworthBandPassFilter(data, lowcut=high_cutoff, highcut=low_cutoff, order=order)
#     print(f"Band-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     filter_instance.PlotFilter(data, band_passed_data, title="Band-Pass_Filter_Result", savepath=savepath)
#     # Take abs of signal
#     band_passed_data = np.abs(band_passed_data)
#     # # notch
#     # start_time = time.time()
#     # notch_filtered_data = filter_instance.NotchFilter(band_passed_data, notch_freq=notch_freq)
#     # print(f"Notch filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(band_passed_data, notch_filtered_data, title="Notch_After_Bandpass_Result", savepath=savepath)
#     # # Extreme lowpass
#     # start_time = time.time()
#     # final_filtered_data = filter_instance.ButterworthLowPass(notch_filtered_data, cutoff=20, order=order)
#     # print(f"Low-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(notch_filtered_data, final_filtered_data, title="Final_Filtered_Result", savepath=savepath)
#     # RMS on data both with extreme lowpass and without
#     # rms_value = filter_instance.Windowed_RMS(final_filtered_data, window_size=100)
#     rms_value_2 = filter_instance.Windowed_RMS(band_passed_data, window_size=100)
#     #plot RMS signal
#     plt.figure(figsize=(12, 4))
#     # plt.plot(rms_value, label='RMS of Fully Filtered Signal', color='blue')
#     plt.plot(rms_value_2, label='RMS of Notch + Bandpass Signal', color='orange', alpha=0.7)
#     plt.title("Windowed RMS Comparison")
#     plt.xlabel("Window Index")
#     plt.ylabel("RMS Value")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(savepath + "RMS_final.png")
#     plt.show()
    
#     # FFT of fully filtered signal
#     freqs, magnitude = filter_instance.FFT(band_passed_data, plot=True)

#     # # Lowpass filter
#     # start_time = time.time()
#     # low_passed_data = filter_instance.ButterworthLowPass(filter_instance.ButterworthHighPass(data, cutoff=20, order=order), cutoff=10, order=order)
#     # print(f"Low-pass filter processing time: {time.time() - start_time:.4f} seconds")
#     # filter_instance.PlotFilter(data, low_passed_data, title="Low-Pass_Filter_Result", savepath=savepath)



class rt_filtering:
    def __init__(self, sample_rate, lp_cutoff=450, hp_cutoff=20, order=4, mains=50.0, notch_bw=1.0):
        self.fs = sample_rate       # Sample rate in Hz
        self.nyq = 0.5 * self.fs    # Nyquist frequency

        # --- design filters (SOS) ---
        self.lp_sos = butter(order, lp_cutoff / self.nyq, btype='lowpass', output='sos')
        self.lp_zi  = sosfilt_zi(self.lp_sos)

        self.hp_sos = butter(order, hp_cutoff / self.nyq, btype='highpass', output='sos')
        self.hp_zi  = sosfilt_zi(self.hp_sos)

        self.bandpass_sos = butter(order, [hp_cutoff / self.nyq, lp_cutoff / self.nyq], btype='bandpass', output='sos')
        self.bandpass_zi  = sosfilt_zi(self.bandpass_sos)

        self.bandstop_sos = butter(order, [ (mains-2)/self.nyq, (mains+2)/self.nyq ], btype='bandstop', output='sos')
        self.bandstop_zi  = sosfilt_zi(self.bandstop_sos)

        # Notch (use fs= to specify Hz directly)
        Q = mains / notch_bw
        b_notch, a_notch = iirnotch(mains, Q, fs=self.fs)
        self.notch_sos = tf2sos(b_notch, a_notch)
        self.notch_zi  = sosfilt_zi(self.notch_sos)

    def Windowed_RMS(self, filtered_data, window_size=50):
        rms_values = []
        for i in range(0, len(filtered_data), window_size):
            window = filtered_data[i:i+window_size]
            rms = np.sqrt(np.mean(np.square(np.abs(window))))
            rms_values.append(rms)
        return rms_values

    # --- Chunk processing ---
    def process_chunk(self, chunk):
        x = np.asarray(chunk, dtype=float)
        if x.size < 50:
            return [], x
        start_time = time.time()
        # Bandpass
        y_bandpass,self.bandpass_zi = sosfilt(self.bandpass_sos, x,  zi=self.bandpass_zi)
        # Windowed RMS
        rms_values = self.Windowed_RMS(y_bandpass, window_size=50)
        #print(f"Chunk processing time: {time.time() - start_time:.4f} seconds")

        return y_bandpass, rms_values
    

if __name__ == "__main__":
    sample_rate = 2148  # Hz
    # Create an instance of the real-time filtering class
    rt_filter_instance = rt_filtering(sample_rate, lp_cutoff=300, hp_cutoff=20, order=2)
    # define read and save paths
    readpath = "./Outputs/RecordedEMG/"
    savepath = "./Outputs/AttemptedFiltering/"
    # Read recorded EMG data from a CSV file
    data = np.loadtxt(readpath + "EMG_FastMovement.csv", delimiter=",")
    chunk_size = 50  # samples
    all_rms = []
    all_filtered = []
    for start in range(0, len(data), chunk_size):
        chunk = data[start:start+chunk_size]
        filtered, rms_values = rt_filter_instance.process_chunk(chunk)
        all_rms.extend(rms_values)
        all_filtered.extend(filtered)
    # Plot RMS and bandpass of all chunks, and original signal
    plt.figure(figsize=(12, 6))
    plt.subplot(3,1,1)
    plt.plot(data, label='Original Data', alpha=0.7)
    plt.title("Original Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.subplot(3,1,2)
    plt.plot(all_filtered, label='Band-Passed Signal', color='green')
    plt.title("Band-Passed Signal (Real-Time Processing)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(all_rms, label='RMS of Band-Passed Signal', color='orange')
    plt.title("Windowed RMS of Band-Passed Signal (Real-Time Processing)")
    plt.xlabel("Window Index")
    plt.ylabel("RMS Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath + "RMS_Band-Pass_Signal_RT.png")
    plt.show()