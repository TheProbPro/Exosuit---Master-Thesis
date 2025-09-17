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
from pytrigno import pytrigno # https://delsyseurope.com/downloads/USERSGUIDE/trigno/sdk.pdf https://github.com/axopy/pytrigno
import signal
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "EMG_FastMovement.csv"


# # Signal handler function
# def signal_handler(signal, frame):
#     print('Code stopped')
#     if not TEST:
#         dev.stop()  # Stop the device
#     sys.exit(0)  # Exit the program
    
# desired_hz = 1000  # The desired frequency in Hz
# delay = 1.0 / desired_hz  # Calculate the delay in seconds
# TIME = 0.0
# TEST = 0 # Determines if we send real data, or fake data

# # Register the signal handler
# signal.signal(signal.SIGINT, signal_handler)

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
    emg = DelsysEMG(channel_range=(0, 0), samples_per_read=1, host='localhost', units='mV')
    emg.start()
    data = np.array([])
    TIME = time.time()
    while time.time() - TIME < 10:
        reading = emg.read()
        data = np.append(data, reading)
    emg.stop()

    plt.plot(data)
    plt.savefig("EMG_test.png")
    plt.show()

# # take arguments from the command line
# if __name__ == "__main__":
#     dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=1,
#                              host='localhost', units='mV')

#     # test single-channel
#     dev.start()
    
#     data = np.array([])

#     TIME = time.time()
#     while time.time() - TIME < 10:
#         reading = dev.read()
#         data = np.append(data, reading)
    
#     plt.plot(data)
#     plt.savefig("EMG_test.png")
#     plt.show()
    

#     # Save data to CSV file
#     np.savetxt(FILENAME, data, delimiter=",")


#     dev.stop()
    
#     # # take ip and port from the command line and file name
#     # # run progra for 2 minutes
#     # TIME = time.time()

#     # dev = pytrigno.TrignoEMG(channel_range=(0, 0), host='localhost', samples_per_read=1, units='mV')
#     # dev.start()
#     # data = []

#     # # append readings to list for 10 seconds
#     # while time.time() - TIME < 10:
#     #     print(dev.read())
#     #     #data.append(dev.read())
    
#     # print(data)

#     # # plot the data
#     # plt.plot(data)
#     # plt.show()

    
#     # dev.stop()

    

    

exit()