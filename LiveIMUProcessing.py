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
        try:
            output_queue.put_nowait(data)
        except queue.Full:
            output_queue.get_nowait()  # discard oldest
            output_queue.put_nowait(data)

def Read_EMG(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_emg()
        try:
            output_queue.put_nowait(data)
        except queue.Full:
            output_queue.get_nowait()  # discard oldest
            output_queue.put_nowait(data)

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
    sample_rate_imu = 148 # Hz, for IMU, set in DelsysEMGIMU constructor
    sample_rate_emg = 1248 # Hz, for EMG, set in DelsysEMGIMU constructor

    # Create output queues
    imu_queue = queue.Queue(maxsize=5)
    emg_queue = queue.Queue(maxsize=5)

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
   
    print("Press enter and keep arm still for gyroscope bias estimation. Duration 1 s...")
    input()  # wait for user to press enter
    g_u = []
    g_l = []
    start_time = time.time()
    while time.time() - start_time < 1.0:
        if not imu_queue.empty():
            s = np.asarray(imu_queue.get(), dtype=float).reshape(-1)
            g_u.append(s[3:6])
            g_l.append(s[12:15])
    
    gyro_bias_u = np.mean(np.deg2rad(np.vstack(g_u)), axis=0)
    gyro_bias_l = np.mean(np.deg2rad(np.vstack(g_l)), axis=0)
    print("Gyro bias upper (rad/s):", gyro_bias_u)
    print("Gyro bias lower (rad/s):", gyro_bias_l)

    # Calculate zeroing baseline for elbow angle estimation (assuming straight arm during the first second)
    print("Press enter and keep arm straight for 1 second to calculate zeroing baseline for elbow angle estimation...")
    input()  # wait for user to press enter
    madgwick_u = Madgwick(frequency=sample_rate_imu, beta=0.02)
    madgwick_l = Madgwick(frequency=sample_rate_imu, beta=0.02)
    q_u = np.array([1.0, 0.0, 0.0, 0.0])
    q_l = np.array([1.0, 0.0, 0.0, 0.0])
    zero_samples = []
    hinge_axis_samples = []
    start_time = time.time()
    while time.time() - start_time < 1.0:
        if not imu_queue.empty():
            s = np.asarray(imu_queue.get(), dtype=float).reshape(-1)

            # Extract the upper and lower arm IMU data
            acc_u = s[0:3]
            gyro_u = s[3:6]
            mag_u = s[6:9]
            acc_l = s[9:12]
            gyro_l = s[12:15]
            mag_l = s[15:18]

            # Convert gyro to rad/s and subtract bias
            gyro_u_rad = np.deg2rad(gyro_u) - gyro_bias_u
            gyro_l_rad = np.deg2rad(gyro_l) - gyro_bias_l

            # Normalize accelerometer data
            na_u = np.linalg.norm(acc_u)
            na_l = np.linalg.norm(acc_l)

            if na_u < 1e-6 or na_l < 1e-6:
                # skip update, keep last quaternions
                continue

            acc_u_n = acc_u / na_u
            acc_l_n = acc_l / na_l

            # Update quaternions using Madgwick filter
            q_u_new = madgwick_u.updateIMU(q_u, gyr=gyro_u_rad, acc=acc_u_n)
            q_l_new = madgwick_l.updateIMU(q_l, gyr=gyro_l_rad, acc=acc_l_n)
            if q_u_new is not None:
                q_u = q_u_new
            if q_l_new is not None:
                q_l = q_l_new

            # calculate the relative quaternion
            q_rel = quat_mul(quat_conj(q_u), q_l)  # q_u^{-1} ⊗ q_l  (unit quats)

            # Axis-angle from relative quaternion
            w = np.clip(q_rel[0], -1.0, 1.0)
            angle_rad = 2.0*np.arccos(w)
            #print(f"angle_rad = {angle_rad:.4f} radians, {np.degrees(angle_rad):.2f} degrees")
            sin_half = np.sqrt(max(1.0 - w*w, 1e-12))
            axis = q_rel[1:] / sin_half

            # Calculate signed hinge axis angle - This is casual, not completely simmilar to the previous code
            hinge_axis_samples.append(axis.copy())
            hinge_axis = np.mean(np.vstack(hinge_axis_samples), axis=0)
            hinge_axis /= np.linalg.norm(hinge_axis)

            sign = np.sign(axis @ hinge_axis)
            elbow_flex_deg = np.degrees(angle_rad * sign)

            # add to zeroing samples
            zero_samples.append(elbow_flex_deg)
    # Calculate zeroing baseline
    zero = np.mean(zero_samples)


    print("press enter to start real time elbow angle estimation for 10 seconds...")
    input()  # wait for user to press enter
    q_u = np.array([1.0, 0.0, 0.0, 0.0])
    q_l = np.array([1.0, 0.0, 0.0, 0.0])
    angle_deg = []
    Q_u = []
    Q_l = []
    q = []
    hinge_axis_samples = []
    plot_unsigned_angle = []
    plot_elbow_flex_deg = []
    
    start_time = time.time()
    while time.time() - start_time < 10.0:  # run for 10 seconds
        if not imu_queue.empty():
            # Convert the last input in the queue to numpy array
            s = np.asarray(imu_queue.get(), dtype=float).reshape(-1)
            # Append to the respective lists for logging/plotting
            imu_acc_x.append(s[0])
            imu_acc_y.append(s[1])
            imu_acc_z.append(s[2])
            imu_gyro_x.append(s[3])
            imu_gyro_y.append(s[4])
            imu_gyro_z.append(s[5])
            imu_mag_x.append(s[6])
            imu_mag_y.append(s[7])
            imu_mag_z.append(s[8])
            imu_acc_x_1.append(s[9])
            imu_acc_y_1.append(s[10])
            imu_acc_z_1.append(s[11])
            imu_gyro_x_1.append(s[12])
            imu_gyro_y_1.append(s[13])
            imu_gyro_z_1.append(s[14])
            imu_mag_x_1.append(s[15])
            imu_mag_y_1.append(s[16])
            imu_mag_z_1.append(s[17])

            # Extract the upper and lower arm IMU data
            acc_u = s[0:3]
            gyro_u = s[3:6]
            mag_u = s[6:9]
            acc_l = s[9:12]
            gyro_l = s[12:15]
            mag_l = s[15:18]

            # Convert gyro to rad/s and subtract bias
            gyro_u_rad = np.deg2rad(gyro_u) - gyro_bias_u
            gyro_l_rad = np.deg2rad(gyro_l) - gyro_bias_l

            # Normalize accelerometer data
            na_u = np.linalg.norm(acc_u)
            na_l = np.linalg.norm(acc_l)

            if na_u < 1e-6 or na_l < 1e-6:
                # skip update, keep last quaternions
                angle_deg.append(elbow_angle_from_quats(q_u, q_l))
                continue

            acc_u_n = acc_u / na_u
            acc_l_n = acc_l / na_l

            # Update quaternions using Madgwick filter
            q_u_new = madgwick_u.updateIMU(q_u, gyr=gyro_u_rad, acc=acc_u_n)
            q_l_new = madgwick_l.updateIMU(q_l, gyr=gyro_l_rad, acc=acc_l_n)
            if q_u_new is not None:
                q_u = q_u_new
            if q_l_new is not None:
                q_l = q_l_new
            
            # For logging/plotting
            Q_u.append(q_u.copy())
            Q_l.append(q_l.copy())
            q.append((q_u, q_l))

            # calculate the relative quaternion
            q_rel = quat_mul(quat_conj(q_u), q_l)  # q_u^{-1} ⊗ q_l  (unit quats)

            # Axis-angle from relative quaternion
            w = np.clip(q_rel[0], -1.0, 1.0)
            angle_rad = 2.0*np.arccos(w)
            plot_unsigned_angle.append(np.degrees(angle_rad))
            #print(f"angle_rad = {angle_rad:.4f} radians, {np.degrees(angle_rad):.2f} degrees")
            sin_half = np.sqrt(max(1.0 - w*w, 1e-12))
            axis = q_rel[1:] / sin_half

            # Calculate signed hinge axis angle - This is casual, not completely simmilar to the previous code
            hinge_axis_samples.append(axis.copy())
            hinge_axis = np.mean(np.vstack(hinge_axis_samples), axis=0)
            hinge_axis /= np.linalg.norm(hinge_axis)

            sign = np.sign(axis @ hinge_axis)
            elbow_flex_deg = np.degrees(angle_rad * sign)
            elbow_flex_deg -= zero  # apply zeroing baseline
            # clip to between 0 and 140 degrees, which is the expected range for elbow flexion
            elbow_flex_deg = np.clip(elbow_flex_deg, 0.0, 140.0)
            plot_elbow_flex_deg.append(elbow_flex_deg)

    print("Finished real-time processing. Stopping threads and sensor...")
    stop_event.set()
    emg_imu.stop()
    t_emg.join(timeout=2.0)
    t_imu.join(timeout=2.0)
    emg_imu.stop()

    # Plot the elbow angle over time
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(plot_unsigned_angle, label='Unsigned Elbow Angle (degrees)')
    plt.title("Elbow Angle Over Time")
    plt.xlabel("Time (samples)")
    plt.ylabel("Angle (degrees)")
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(plot_elbow_flex_deg, label='Signed Elbow Flexion (degrees)')
    plt.xlabel("Time (samples)")
    plt.ylabel("Flexion Angle (degrees)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()