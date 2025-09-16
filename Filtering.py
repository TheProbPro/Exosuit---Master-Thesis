import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
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
            plt.xlim(0, self.fs/2)                   # Nyquist limit
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("FFT Magnitude Spectrum of sEMG signal")
            plt.grid(True)
            plt.show()

        return freqs, magnitude

    def Windowed_RMS(self, filtered_data, window_size=100):
        rms_values = []
        for i in range(0, len(filtered_data), window_size):
            window = filtered_data[i:i+window_size]
            rms = np.sqrt(np.mean(np.square(window)))
            rms_values.append(rms)
        return rms_values

if __name__ == "__main__":
    # Sample rate
    sample_rate = 2148  # Hz
    # Cutoff Fequencies
    #low_cutoff = 10  # Hz
    #high_cutoff = 20  # Hz
    high_cutoff = 20
    low_cutoff = 300
    notch_freq = 50  # Hz
    order = 6  # Order of the filter

    # Create an instance of the filtering class
    filter_instance = filtering(sample_rate)

    # define read and save paths
    readpath = "./RecordedEMG/"
    savepath = "./AttemptedFiltering/"
    # Read recorded EMG data from a CSV file
    data = np.loadtxt(readpath + "EMG_SlowMovement.csv", delimiter=",")

    # Calculate FFT of original signal
    freqs, magnitude = filter_instance.FFT(data, plot=True)

    # Apply filters, plot results and check processing time
    # High-pass filter
    start_time = time.time()
    high_passed_data = filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order)
    print(f"High-pass filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, high_passed_data, title="High-Pass_Filter_Result", savepath=savepath)

    # Low-pass filter
    start_time = time.time()
    low_passed_data = filter_instance.ButterworthLowPass(data, cutoff=low_cutoff, order=order)
    print(f"Low-pass filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, low_passed_data, title="Low-Pass_Filter_Result", savepath=savepath)

    # Notch filter
    start_time = time.time()
    notch_filtered_data = filter_instance.NotchFilter(data, notch_freq=notch_freq)
    print(f"Notch filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, notch_filtered_data, title="Notch_Filter_Result", savepath=savepath)
    
    # Combined high-pass and low-pass filter
    start_time = time.time()
    combined_filtered_data = filter_instance.ButterworthLowPass(
        filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order), cutoff=low_cutoff, order=order)
    print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, combined_filtered_data, title="Highpass_Lowpass_Result", savepath=savepath)

    # Combined low-pass and high-pass filter
    start_time = time.time()
    combined_filtered_data_2 = filter_instance.ButterworthHighPass(
        filter_instance.ButterworthLowPass(data, cutoff=low_cutoff, order=order), cutoff=high_cutoff, order=order)
    print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, combined_filtered_data_2, title="Lowpass_Highpass_Result", savepath=savepath)

    # Combined High-pass, Notch, and low-pass filter
    start_time = time.time()
    combined_filtered_data_3 = filter_instance.ButterworthLowPass(
        filter_instance.NotchFilter(
            filter_instance.ButterworthHighPass(data, cutoff=high_cutoff, order=order), notch_freq=notch_freq), cutoff=low_cutoff, order=order)
    print(f"Combined filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, combined_filtered_data_3, title="Highpass_Notch_Lowpass_Result", savepath=savepath)

    # band-pass filter
    start_time = time.time()
    band_passed_data = filter_instance.ButterworthBandPassFilter(data, lowcut=high_cutoff, highcut=low_cutoff, order=order)
    print(f"Band-pass filter processing time: {time.time() - start_time:.4f} seconds")
    filter_instance.PlotFilter(data, band_passed_data, title="Band-Pass_Filter_Result", savepath=savepath)

    # RMS on band-passed data
    rms_value = filter_instance.Windowed_RMS(band_passed_data, window_size=100)
    #plot RMS signal
    plt.figure(figsize=(12, 4))
    plt.plot(rms_value, label='RMS of Band-Passed Signal', color='green')
    plt.title("Windowed RMS of Band-Passed Signal")
    plt.xlabel("Window Index")
    plt.ylabel("RMS Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath + "RMS_Band-Pass_Signal.png")
    plt.show()
    



