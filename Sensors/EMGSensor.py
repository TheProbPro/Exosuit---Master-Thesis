# Before startup

# 1. Connect EMG sensors and start Trigno Control Utility

##################################
# 2.       Disable firewall      #
##################################

# 3. Start client side

# 4. Start server side

import socket
import time
import csv
import ast
import sys
from Sensors.pytrigno import pytrigno # https://delsyseurope.com/downloads/USERSGUIDE/trigno/sdk.pdf https://github.com/axopy/pytrigno
import signal
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "EMG_FastMovement.csv"

# TODO: Check if this is correct. There might be issues, and figure out how to check sample rate from the sensor in the code
class DelsysEMGIMU:
    def __init__(self,
                 emg_channel_range=(0, 0),
                 imu_channel_range=(0, 0),
                 emg_samples_per_read=1,
                 imu_samples_per_read=1,
                 host='localhost',
                 emg_units='mV'):
        """
        EMG is 2000 Hz; Accel (IMU) is ~148.1 Hz.
        """
        self.host = host
        self.emg_units = emg_units

        # Devices
        self.emg_dev = pytrigno.TrignoEMG(
            channel_range=emg_channel_range,
            samples_per_read=emg_samples_per_read,
            host=self.host,
            units=self.emg_units
        )
        # self.imu_dev = pytrigno.TrignoAccel(
        #     channel_range=imu_channel_range,
        #     samples_per_read=imu_samples_per_read,
        #     host=self.host
        # )
        self.imu_dev = pytrigno.TrignoIMU(
            channel_range=imu_channel_range,
            samples_per_read=imu_samples_per_read,
            host=self.host
        )

        self.is_running = False
        self.emg_channel_range = emg_channel_range
        self.imu_channel_range = imu_channel_range

    def start(self):
        if not self.is_running:
            self.emg_dev.start()
            #self.imu_dev.start()
            self.is_running = True

    def read_emg(self):
        if not self.is_running:
            raise RuntimeError("Device not started. Call start() before read().")
        return self.emg_dev.read()   # shape: (n_emg_channels, emg_samples_per_read)

    def read_imu(self):
        if not self.is_running:
            raise RuntimeError("Device not started. Call start() before read().")
        return self.imu_dev.read()   # shape: (n_imu_channels*3, imu_samples_per_read) (x/y/z stacked)

    def read(self):
        """
        Read both streams once.
        Returns: {"emg": np.ndarray, "accel": np.ndarray}
        """
        if not self.is_running:
            raise RuntimeError("Device not started. Call start() before read().")
        emg = self.emg_dev.read()
        accel = self.imu_dev.read()
        return {"emg": emg, "accel": accel}

    def stop(self):
        if self.is_running:
            # Stop in reverse order just to be neat
            #self.imu_dev.stop()
            self.emg_dev.stop()
            self.is_running = False

    # Channel management
    def set_emg_channel_range(self, channel_range):
        if self.is_running:
            raise RuntimeError("Cannot change EMG channel range while running.")
        self.emg_channel_range = channel_range
        self.emg_dev.set_channel_range(channel_range)

    def set_imu_channel_range(self, channel_range):
        if self.is_running:
            raise RuntimeError("Cannot change IMU channel range while running.")
        self.imu_channel_range = channel_range
        self.imu_dev.set_channel_range(channel_range)


class DelsysEMG:
    def __init__(self, channel_range=(0, 0), samples_per_read=1,
                 host='localhost', port=50041, units='mV'):
        self.channel_range = channel_range
        self.samples_per_read = samples_per_read
        self.host = host
        self.port = port
        self.units = units
        self.dev = pytrigno.TrignoEMG(channel_range=self.channel_range,
                                      samples_per_read=self.samples_per_read,
                                      host=self.host, units=self.units)
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.dev.start()
            self.is_running = True

    def read(self):
        if self.is_running:
            return self.dev.read()
        else:
            raise RuntimeError("Device not started. Call start() before read().")

    def stop(self):
        if self.is_running:
            self.dev.stop()
            self.is_running = False

    def set_channel_range(self, channel_range):
        if not self.is_running:
            self.channel_range = channel_range
            self.dev.set_channel_range(channel_range)
        else:
            raise RuntimeError("Cannot change channel range while device is running.")




if __name__ == "__main__":
    # emg = DelsysEMG(channel_range=(0, 0), samples_per_read=1, host='localhost', units='mV')
    # emg.start()
    # data = np.array([])
    # TIME = time.time()
    # while time.time() - TIME < 10:
    #     reading = emg.read()
    #     data = np.append(data, reading)
    # emg.stop()

    # print(len(data))
    # plt.plot(data)
    # plt.savefig("EMG_test.png")
    # plt.show()

    # Use EMG and IMU class
    dev = DelsysEMGIMU(
        emg_channel_range=(0, 15),      # your EMG sensors
        imu_channel_range=(0, 15),      # matching IMU/Accel sensors
        emg_samples_per_read=1200,      # 1 s EMG chunks
        imu_samples_per_read=140        # ~1 s accel chunks
    )
    dev.start()

    data = dev.read()
    emg = data["emg"]          # shape: (n_emg_channels, 2000)
    accel = data["accel"]      # shape: (n_axes, 148) where axes=channels*3
    print("EMG shape: ", emg.shape)
    print("Accel shape: ", accel.shape)

    # plot emg and accel from first sensor
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(emg[0, :], label='EMG Channel 0', color='blue')
    plt.title('EMG Signal from Channel 0')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(accel[0:3, :].T)  # plot x, y, z
    plt.title('Accelerometer Signal from Channel 0')
    plt.xlabel('Samples')
    plt.ylabel('Acceleration (g)')
    plt.legend(['Acc.X', 'Acc.Y', 'Acc.Z'])
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(accel[3:6, :].T)  # plot gyro x, y, z
    plt.title('Gyroscope Signal from Channel 0')
    plt.xlabel('Samples')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.legend(['Gyro.X', 'Gyro.Y', 'Gyro.Z'])
    plt.grid()
    plt.tight_layout()
    plt.show()


    dev.stop()