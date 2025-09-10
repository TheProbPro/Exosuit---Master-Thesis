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


# Signal handler function
def signal_handler(signal, frame):
    print('Code stopped')
    if not TEST:
        dev.stop()  # Stop the device
    sys.exit(0)  # Exit the program
    
desired_hz = 1000  # The desired frequency in Hz
delay = 1.0 / desired_hz  # Calculate the delay in seconds
TIME = 0.0
TEST = 0 # Determines if we send real data, or fake data

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def Test():
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=270,
                             host='localhost', units='mV')

    # test single-channel
    dev.start()
    for i in range(4):
        data = dev.read()
        assert data.shape == (1, 270)
        print(data.max())
    dev.stop()

    # test multi-channel
    dev.set_channel_range((0, 4))
    dev.start()
    for i in range(4):
        data = dev.read()
        assert data.shape == (5, 270)
        print(data.max())
    dev.stop()



# take arguments from the command line
if __name__ == "__main__":
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=1,
                             host='localhost', units='mV')

    # test single-channel
    dev.start()
    
    data = np.array([])

    TIME = time.time()
    while time.time() - TIME < 10:
        reading = dev.read()
        data = np.append(data, reading)
    
    plt.plot(data)
    plt.savefig("EMG_FastMovement.png")
    plt.show()
    

    # Save data to CSV file
    np.savetxt(FILENAME, data, delimiter=",")


    dev.stop()
    
    # # take ip and port from the command line and file name
    # # run progra for 2 minutes
    # TIME = time.time()

    # dev = pytrigno.TrignoEMG(channel_range=(0, 0), host='localhost', samples_per_read=1, units='mV')
    # dev.start()
    # data = []

    # # append readings to list for 10 seconds
    # while time.time() - TIME < 10:
    #     print(dev.read())
    #     #data.append(dev.read())
    
    # print(data)

    # # plot the data
    # plt.plot(data)
    # plt.show()

    
    # dev.stop()

    

    

exit()