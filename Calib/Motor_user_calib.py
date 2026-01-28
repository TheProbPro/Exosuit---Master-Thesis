from Motors.DynamixelHardwareInterface import Motors

import time
import os
import pandas as pd

USER_NAME = 'VictorBNielsen'
SAVE_PATH = 'Calib/Users/{}'.format(USER_NAME)
FILE_NAME = 'motor_calib_data.csv'

if __name__ == "__main__":
    motor = Motors(baudrate=4500000)
    motor.disable_torque()
    time.sleep(1)

    print("Fully extend your arm and then press Enter to record position.")
    input()
    extended_position = motor.get_position(motor_id=motor.motor_ids[0])
    print(f"Recorded extended position: {extended_position}")
    Extended_elbow = extended_position
    print(f"Extended elbow (motor 0): {Extended_elbow}")

    motor.enable_torque()
    time.sleep(1)

    current = motor.torq2curcom(1.0)
    motor.sendMotorCommand(motor.motor_ids[0], current)

    print("Fully flex your arm, and wait for the cable to gain tension, and then press Enter to record position.")
    input()
    motor.sendMotorCommand(motor.motor_ids[0], 0.0)
    flexed_position = motor.get_position(motor_id=motor.motor_ids[0])
    print(f"Recorded flexed position: {flexed_position}")
    Flexed_elbow = flexed_position
    print(f"Flexed elbow (motor 0): {Flexed_elbow}")

    motor.disable_torque()
    time.sleep(1)

    print(f"Saving data to userfile: {SAVE_PATH}/{FILE_NAME}")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    df = pd.DataFrame({
        'Flexed': [Flexed_elbow],
        'Extended': [Extended_elbow]
    })
    df.to_csv(f'{SAVE_PATH}/{FILE_NAME}', index=False, mode='w')
    print("Data saved successfully.")
    
    print("Closing down...")
    
    motor.close()