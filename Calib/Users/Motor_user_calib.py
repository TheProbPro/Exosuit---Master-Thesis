from Motors.DynamixelHardwareInterface import Motors

import time
import os
import pandas as pd

USER_NAME = 'VictorBNielsen'
SAVE_PATH = 'Calib/Users/{}'.format(USER_NAME)
FILE_NAME = 'motor_calib_data.csv'

if __name__ == "__main__":
    motor = Motors()
    motor.disable_torque()
    time.sleep(1)

    print("Fully flex your arm and then press Enter to record position.")
    input()
    flexed_position = motor.get_position()
    print(f"Recorded flexed position: {flexed_position}")

    print("Fully extend your arm and then press Enter to record position.")
    input()
    extended_position = motor.get_position()
    print(f"Recorded extended position: {extended_position}")

    print(f"Saving data to userfile: {SAVE_PATH}/{FILE_NAME}")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    df = pd.DataFrame({
        'Flexed': flexed_position,
        'Extended': extended_position
    })
    df.to_csv(f'{SAVE_PATH}/{FILE_NAME}', index=False)
    print("Data saved successfully.")
    
    print("Closing down...")
    
    motor.disconnect()