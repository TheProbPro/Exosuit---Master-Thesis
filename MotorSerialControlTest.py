from Motors.DynamixelHardwareInterface import Motors

import time
import os
import pandas as pd

if __name__ == "__main__":
    motor = Motors(baudrate=4500000)
    time.sleep(1)

    #motor.changeBaudrate()
    #time.sleep(1)

    motor.disable_torque()
    time.sleep(1)

    motor.close()