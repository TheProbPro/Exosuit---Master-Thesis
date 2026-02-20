from Sensors.EMGSensor import DelsysEMGIMU, DelsysEMG
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue
import signal

n_Sensors = 1

# First 3 channles of IMU is accel, next 3 is gyro, last 3 is magnetometer.

stop_event = threading.Event()

def Read_IMU(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_imu()
        output_queue.put(data)
    
def Read_EMG(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_emg()
        output_queue.put(data)

def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create output queues
    imu_queue = queue.Queue()
    emg_queue = queue.Queue()

    # Create EMG-IMU sensor instance and start it
    emg_imu = DelsysEMGIMU(emg_channel_range=(0,n_Sensors-1), imu_channel_range=(0,(9*n_Sensors)-1), emg_samples_per_read=1, imu_samples_per_read=1, host='localhost', emg_units='mV')
    emg_imu.start()
    time.sleep(1.0)

    # perform 1 read to test
    # data_read = emg_imu.read()
    # print("EMG Data Shape:", data_read["emg"].shape)
    # print("IMU Data Shape:", data_read["imu"].shape)

    # i = 0
    # for data in data_read["emg"]:
    #     i += 1
    #     print("EMG Data max:", max(data), " at idx: ", i)

    # i = 0
    # for data in data_read["imu"]:
    #     i += 1
    #     print("IMU Data max:", max(data), " at idx: ", i)

    # perform 1 read of each to test streaming
    # emg = emg_imu.read_emg()
    # imu = emg_imu.read_imu()

    # print("EMG Data Shape:", emg.shape)
    # print("IMU Data Shape:", imu.shape)

    t_emg = threading.Thread(target=Read_EMG, args=(emg_imu, emg_queue))
    t_imu = threading.Thread(target=Read_IMU, args=(emg_imu, imu_queue))
    t_emg.start()
    t_imu.start()

    # perform 1 second read to test
    start = time.time()
    emg = []
    imu_acc_x = []
    imu_acc_y = []
    imu_acc_z = []
    imu_gyro_x = []
    imu_gyro_y = []
    imu_gyro_z = []
    imu_mag_x = []
    imu_mag_y = []
    imu_mag_z = []
    # while True:
    #     if time.time() - start > 1.0:
    #         stop_event.set()
    #         break
    time.sleep(1.0)
    stop_event.set()
    
    # Empty the queues
    for x in emg_queue.queue:
        emg.append(x[0])
    
    for x in imu_queue.queue:
        imu_acc_x.append(x[0])
        imu_acc_y.append(x[1])
        imu_acc_z.append(x[2])
        imu_gyro_x.append(x[3])
        imu_gyro_y.append(x[4])
        imu_gyro_z.append(x[5])
        imu_mag_x.append(x[6])
        imu_mag_y.append(x[7])
        imu_mag_z.append(x[8])

        # data_read = emg_imu.read()
        # emg.append(data_read["emg"][0])
        # imu_acc_x.append(data_read["imu"][0])
        # imu_acc_y.append(data_read["imu"][1])
        # imu_acc_z.append(data_read["imu"][2])
        # imu_gyro_x.append(data_read["imu"][3])
        # imu_gyro_y.append(data_read["imu"][4])
        # imu_gyro_z.append(data_read["imu"][5])
        # imu_mag_x.append(data_read["imu"][6])
        # imu_mag_y.append(data_read["imu"][7])
        # imu_mag_z.append(data_read["imu"][8])
    # data_read = emg_imu.read_imu()
    # imu_acc_x = data_read[0]
    # imu_acc_y = data_read[1]
    # imu_acc_z = data_read[2]
    # imu_gyro_x = data_read[3]
    # imu_gyro_y = data_read[4]
    # imu_gyro_z = data_read[5]
    # imu_mag_x = data_read[6]
    # imu_mag_y = data_read[7]
    # imu_mag_z = data_read[8]
    
    print("EMG Data Shape:", np.array(emg).shape)
    print("IMU Data Shape:", np.array([imu_acc_x, imu_acc_y, imu_acc_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z]).shape)

    stop_event.set()
    t_emg.join()
    t_imu.join()
    emg_imu.stop()
    
    # plot the 1 second of imu data and emg data
    plt.figure(figsize=(12, 6))
    plt.plot(emg)
    plt.title("EMG Data Over 1 Second")
    plt.xlabel("Time (samples)")
    plt.ylabel("EMG Signal (mV)")
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(imu_acc_x, label='Accel X')
    plt.plot(imu_acc_y, label='Accel Y')
    plt.plot(imu_acc_z, label='Accel Z')
    plt.title("IMU Accelerometer Data Over 1 Second")
    plt.xlabel("Time (samples)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(imu_gyro_x, label='Gyro X')
    plt.plot(imu_gyro_y, label='Gyro Y')
    plt.plot(imu_gyro_z, label='Gyro Z')
    plt.title("IMU Gyroscope Data Over 1 Second")
    plt.xlabel("Time (samples)")
    plt.ylabel("Angular Velocity (deg/s)")
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(imu_mag_x, label='Mag X')
    plt.plot(imu_mag_y, label='Mag Y')
    plt.plot(imu_mag_z, label='Mag Z')
    plt.title("IMU Magnetometer Data Over 1 Second")
    plt.xlabel("Time (samples)")
    plt.ylabel("Magnetic Field (µT)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
