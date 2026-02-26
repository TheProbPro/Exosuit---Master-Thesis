import numpy as np
from ahrs.filters import Madgwick

class IMUProcessing:
    """
    This class processes the raw IMU data from the upper and lower arm in order to convert it to quaternions and calculate the elbow angle.
    The class uses the Madgwick filter to calculate the quaternions from the accelerometer and gyroscope data. The elbow angle is calculated from the relative orientation of the upper and lower arm quaternions.
    """
    def __init__(self, sample_rate=148, beta=0.02):
        """
        :param sample_rate: Sample rate of the IMU data gathering in Hz (default: 148)
        :param beta: Madgwick filter gain (default: 0.02) raise if the sample rate is low, lower if the sample rate is high. The optimal value depends on the noise level of the IMU data and the dynamics of the motion being measured.
        """
        self.sample_rate = sample_rate
        self.madgwick_upper = Madgwick(sample_rate=sample_rate, beta=beta)
        self.madgwick_lower = Madgwick(sample_rate=sample_rate, beta=beta)
        self.q_upper = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion for upper IMU
        self.q_lower = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion for lower IMU
        self.gyr_bias_upper = [0.0, 0.0, 0.0]
        self.gyr_bias_lower = [0.0, 0.0, 0.0]

    def calculate_bias(self, imu_list: list):
        """
        Calculates the gyroscope bias from a list containing the raw IMU data.
        
        :param imu_list: a list containing the raw IMU data, for the upper and lower arm, in the format [(acc_upper, gyr_upper), (acc_lower, gyr_lower)], where acc is the accelerometer data and gyr is the gyroscope data, over a 1 s window.
        
        returns: the gyroscope bias for the upper and lower arm as a tuple (gyr_bias_upper, gyr_bias_lower)
        """
        g_u = []
        
        g_l = []
        for s in imu_list:
            s = np.asarray(s, dtype=float).reshape(-1)
            g_u.append(s[3:6])
            g_l.append(s[12:15])

        self.gyr_bias_upper = np.mean(np.deg2rad(np.vstack(g_u)), axis=0)
        self.gyr_bias_lower = np.mean(np.deg2rad(np.vstack(g_l)), axis=0)

        return (self.gyr_bias_upper, self.gyr_bias_lower)
        

    def calculate_quarternions(self, acc_upper: np.ndarray, gyr_upper: np.ndarray, acc_lower: np.ndarray, gyr_lower: np.ndarray):
        """
        Calculates the quaternions for the upper and lower arm based on the IMU data from the two EMG sensors, utilizing the Madgwick filter.
        
        :param acc_upper: Accelerometer data for the upper arm as a numpy array of shape (3,) containing the x, y, z accelerations in m/s^2
        :param gyr_upper: Gyroscope data for the upper arm as a numpy array of shape (3,) containing the x, y, z angular velocities in degrees/s
        :param acc_lower: Accelerometer data for the lower arm as a numpy array of shape (3,) containing the x, y, z accelerations in m/s^2
        :param gyr_lower: Gyroscope data for the lower arm as a numpy array of shape (3,) containing the x, y, z angular velocities in degrees/s
        
        returns: a tuple containing the quaternions for the upper and lower arm as numpy arrays of shape (4,) in the format [w, x, y, z]
        """
        # Convert gyroscope data from degrees/s to radians/s and apply bias correction
        gyr_upper_rad = np.deg2rad(gyr_upper) - self.gyr_bias_upper
        gyr_lower_rad = np.deg2rad(gyr_lower) - self.gyr_bias_lower

        # Normalize accelerometer data
        na_u =np.linalg.norm(acc_upper)
        na_l = np.linalg.norm(acc_lower)
        if na_u < 1e-6 or na_l < 1e-6:
            return self.q_upper, self.q_lower  # Avoid division by zero, return previous quaternions
        
        acc_upper_normalized = acc_upper / na_u
        acc_lower_normalized = acc_lower / na_l

        # Update quaternions using Madgwick filter
        q_upper_new = self.madgwick_upper.update(self.q_upper, gyr=gyr_upper_rad, acc=acc_upper_normalized)
        q_lower_new = self.madgwick_lower.update(self.q_lower, gyr=gyr_lower_rad, acc=acc_lower_normalized)
        if q_upper_new is not None:
            self.q_upper = q_upper_new
        if q_lower_new is not None:
            self.q_lower = q_lower_new

        return (self.q_upper, self.q_lower)


    def calculate_elbow_angle(self, quat_upper, quat_lower):
        """
        Caclulates the elbow angle in degrees from the quaternions of the upper and lower arm. 
        The angle is calculated as the relative orientation between the two quaternions, and then converted to an axis-angle representation to extract the angle of rotation around the hinge axis of the elbow. 
        The angle is signed based on the direction of rotation around the hinge axis, and zeroed using the first second of data to account for any initial offset when the arm is straight.
        
        :param quat_upper: Quaternion for the upper arm as a numpy array of shape (4,) in the format [w, x, y, z]
        :param quat_lower: Quaternion for the lower arm as a numpy array of shape (4,) in the format [w, x, y, z]

        returns: the elbow angle in degrees as a numpy array of shape (N,) where N is the number of samples, with positive values indicating flexion and negative values indicating extension.
        """

        # Convert lists of quaternions to numpy arrays
        Q_u = np.asarray(quat_upper)
        Q_l = np.asarray(quat_lower)

        # Relative quaternion Q_rel = conj(Q_u) ⊗ Q_l
        Q_u_conj = Q_u.copy()
        Q_u_conj[:, 1:] *= -1.0

        # Quaternion multiplication (conjugate of upper arm quaternion multiplied by lower arm quaternion)
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

        # Estimate axis TODO: This used to be over a window of time, so this might need to be changed to either a calibration step or a known angle we are interested in ex. hinge_axis = np.array([0.0, 1.0, 0.0])  # example: known elbow axis.
        hinge_axis = np.mean(axis, axis=0)
        hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)

        # Signed hinge angle
        sign = np.sign(axis @ hinge_axis)
        elbow_flex_deg = np.degrees(theta * sign)

        # Zero using first second (straight arm)
        zero = np.mean(elbow_flex_deg)
        elbow_flex_deg -= zero

        return elbow_flex_deg