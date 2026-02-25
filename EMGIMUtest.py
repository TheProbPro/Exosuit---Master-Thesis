from Sensors.EMGSensor import DelsysEMGIMU, DelsysEMG
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue
import signal
from ahrs.filters import Madgwick
import pandas as pd

n_Sensors = 2

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

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def elbow_angle_from_quats(q_u, q_l):
    q_rel = quat_mul(quat_conj(q_u), q_l)  # q_u^{-1} ⊗ q_l  (unit quats)
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle_rad = 2.0*np.arccos(w)
    return np.degrees(angle_rad)

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
    imu_acc_x_1 = []
    imu_acc_y_1 = []
    imu_acc_z_1 = []
    imu_gyro_x_1 = []
    imu_gyro_y_1 = []
    imu_gyro_z_1 = []
    imu_mag_x_1 = []
    imu_mag_y_1 = []
    imu_mag_z_1 = []
   
    time.sleep(10.0)
    stop_event.set()
    t_emg.join()
    t_imu.join()
    emg_imu.stop()
    
    # Empty the queues
    for x in emg_queue.queue:
        emg.append(x[0])
    
    q = []
    imu_list = list(imu_queue.queue)  # snapshot
    fs = len(imu_list) / 10.0
    print("fs=", fs)

    # ---- NEW: gyro bias from first 1 second (keep arm still at start) ----
    bias_N = int(1.0 * fs)
    bias_N = max(10, min(bias_N, len(imu_list)))

    g_u = []
    g_l = []
    for s in imu_list[:bias_N]:
        s = np.asarray(s, dtype=float).reshape(-1)
        g_u.append(s[3:6])
        g_l.append(s[12:15])

    gyro_bias_u = np.mean(np.deg2rad(np.vstack(g_u)), axis=0)
    gyro_bias_l = np.mean(np.deg2rad(np.vstack(g_l)), axis=0)
    print("gyro_bias_u (rad/s) =", gyro_bias_u)
    print("gyro_bias_l (rad/s) =", gyro_bias_l)

    madgwick_u = Madgwick(frequency=fs, beta=0.02)
    madgwick_l = Madgwick(frequency=fs, beta=0.02)
    q_u = np.array([1.0, 0.0, 0.0, 0.0])
    q_l = np.array([1.0, 0.0, 0.0, 0.0])
    angle_deg = []
    Q_u = []
    Q_l = []

    print("Processing")

    for x in imu_list:
        imu_acc_x.append(x[0])
        imu_acc_y.append(x[1])
        imu_acc_z.append(x[2])
        imu_gyro_x.append(x[3])
        imu_gyro_y.append(x[4])
        imu_gyro_z.append(x[5])
        imu_mag_x.append(x[6])
        imu_mag_y.append(x[7])
        imu_mag_z.append(x[8])
        imu_acc_x_1.append(x[9])
        imu_acc_y_1.append(x[10])
        imu_acc_z_1.append(x[11])
        imu_gyro_x_1.append(x[12])
        imu_gyro_y_1.append(x[13])
        imu_gyro_z_1.append(x[14])
        imu_mag_x_1.append(x[15])
        imu_mag_y_1.append(x[16])
        imu_mag_z_1.append(x[17])

        x = np.asarray(x, dtype=float).reshape(-1)

        acc_u  = x[0:3]
        gyro_u = x[3:6]
        mag_u  = x[6:9]
        acc_l  = x[9:12]
        gyro_l = x[12:15]
        mag_l  = x[15:18]

        # convert gyro to rad/s instead of deg/s and subtract bias
        gyro_u_rad = np.deg2rad(gyro_u) - gyro_bias_u
        gyro_l_rad = np.deg2rad(gyro_l) - gyro_bias_l

        # Normalize
        na_u = np.linalg.norm(acc_u)
        na_l = np.linalg.norm(acc_l)

        if na_u < 1e-6 or na_l < 1e-6:
            # skip update, keep last quats
            angle_deg.append(elbow_angle_from_quats(q_u, q_l))
            continue

        acc_u_n = acc_u / na_u
        acc_l_n = acc_l / na_l

        q_u_new = madgwick_u.updateIMU(q_u, gyr=gyro_u_rad, acc=acc_u_n)
        q_l_new = madgwick_l.updateIMU(q_l, gyr=gyro_l_rad, acc=acc_l_n)

        # q_u_new = madgwick_u.updateIMU(q_u, gyr=gyro_u_rad, acc=acc_u)
        # q_l_new = madgwick_l.updateIMU(q_l, gyr=gyro_l_rad, acc=acc_l)

        if q_u_new is not None:
            q_u = q_u_new
        if q_l_new is not None:
            q_l = q_l_new

        Q_u.append(q_u.copy())
        Q_l.append(q_l.copy())

        q.append((q_u, q_l))

        angle_deg.append(elbow_angle_from_quats(q_u, q_l))
        
    Q_u = np.asarray(Q_u)  # (N,4)
    Q_l = np.asarray(Q_l)

    # Relative quaternion Q_rel = conj(Q_u) ⊗ Q_l
    Q_u_conj = Q_u.copy()
    Q_u_conj[:, 1:] *= -1.0

    w1,x1,y1,z1 = Q_u_conj.T
    w2,x2,y2,z2 = Q_l.T
    Q_rel = np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    # Axis-angle from relative quaternion
    w = np.clip(Q_rel[:,0], -1.0, 1.0)
    theta = 2.0*np.arccos(w)  # radians
    sin_half = np.sqrt(np.maximum(1.0 - w*w, 1e-12))
    axis = Q_rel[:,1:] / sin_half[:,None]  # (N,3)

    # ---- choose a flex/extend window (edit these if needed) ----
    i0 = int(2.0*fs)
    i1 = int(9.0*fs)

    hinge_axis = np.mean(axis[i0:i1], axis=0)
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)

    # Signed hinge angle
    sign = np.sign(axis @ hinge_axis)
    elbow_flex_deg = np.degrees(theta * sign)

    # Zero using first second (straight arm)
    zero = np.mean(elbow_flex_deg[:int(1.0*fs)])
    elbow_flex_deg -= zero

    # print("EMG Data Shape:", np.array(emg).shape)
    # print("IMU Data Shape:", np.array([imu_acc_x, imu_acc_y, imu_acc_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z]).shape)
    print("Ending program")
    
    
    # plot angle over time
    plt.figure(figsize=(12, 6))
    plt.plot(elbow_flex_deg)
    plt.title("Elbow Angle Over Time")
    plt.xlabel("Time (samples)")
    plt.ylabel("Elbow Angle (degrees)")
    plt.grid()
    plt.show()

    # plot the different axis of the imu signal against eachother
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(imu_acc_x, label='Accel X')
    plt.plot(imu_acc_x_1, label='Accel X 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(imu_acc_y, label='Accel Y')
    plt.plot(imu_acc_y_1, label='Accel Y 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(imu_acc_z, label='Accel Z')
    plt.plot(imu_acc_z_1, label='Accel Z 1')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(imu_gyro_x, label='Gyro X')
    plt.plot(imu_gyro_x_1, label='Gyro X 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(imu_gyro_y, label='Gyro Y')
    plt.plot(imu_gyro_y_1, label='Gyro Y 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(imu_gyro_z, label='Gyro Z')
    plt.plot(imu_gyro_z_1, label='Gyro Z 1')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(imu_mag_x, label='Mag X')
    plt.plot(imu_mag_x_1, label='Mag X 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(imu_mag_y, label='Mag Y')
    plt.plot(imu_mag_y_1, label='Mag Y 1')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(imu_mag_z, label='Mag Z')
    plt.plot(imu_mag_z_1, label='Mag Z 1')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Save the raw imu data to a .csv file
    # imu_data = np.array([imu_acc_x, imu_acc_y, imu_acc_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z,
    #                      imu_acc_x_1, imu_acc_y_1, imu_acc_z_1, imu_gyro_x_1, imu_gyro_y_1, imu_gyro_z_1, imu_mag_x_1, imu_mag_y_1, imu_mag_z_1]).T
    # imu_data = imu_data.reshape(-1, 18)  # Ensure 2D shape for DataFrame
    # imu_df = pd.DataFrame(imu_data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z',
    #                                          'acc_x_1', 'acc_y_1', 'acc_z_1', 'gyro_x_1', 'gyro_y_1', 'gyro_z_1', 'mag_x_1', 'mag_y_1', 'mag_z_1'])
    # imu_df.to_csv('Outputs/RecordedEMG/imu_data.csv', index=False)

    # # plot the 1 second of imu data and emg data
    # plt.figure(figsize=(12, 6))
    # plt.plot(emg)
    # plt.title("EMG Data Over 1 Second")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("EMG Signal (mV)")
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(imu_acc_x, label='Accel X')
    # plt.plot(imu_acc_y, label='Accel Y')
    # plt.plot(imu_acc_z, label='Accel Z')
    # plt.title("IMU Accelerometer Data Over 1 Second")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Acceleration (g)")
    # plt.legend()
    # plt.grid()
    # plt.subplot(3, 1, 2)
    # plt.plot(imu_gyro_x, label='Gyro X')
    # plt.plot(imu_gyro_y, label='Gyro Y')
    # plt.plot(imu_gyro_z, label='Gyro Z')
    # plt.title("IMU Gyroscope Data Over 1 Second")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Angular Velocity (deg/s)")
    # plt.legend()
    # plt.grid()
    # plt.subplot(3, 1, 3)
    # plt.plot(imu_mag_x, label='Mag X')
    # plt.plot(imu_mag_y, label='Mag Y')
    # plt.plot(imu_mag_z, label='Mag Z')
    # plt.title("IMU Magnetometer Data Over 1 Second")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Magnetic Field (µT)")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
