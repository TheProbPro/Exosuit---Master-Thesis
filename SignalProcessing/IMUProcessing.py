import numpy as np
from ahrs.filters import Madgwick

class IMUProcessing:
    """
    This class processes the raw IMU data from the upper and lower arm in order to convert it to quaternions and calculate the elbow angle.
    The class uses the Madgwick filter to calculate the quaternions from the accelerometer and gyroscope data. The elbow angle is calculated from the relative orientation of the upper and lower arm quaternions.
    """
    def __init__(self, sample_rate=148, beta=0.02, theta_min=0.0, theta_max=140.0):
        """
        :param sample_rate: Sample rate of the IMU data gathering in Hz (default: 148)
        :param beta: Madgwick filter gain (default: 0.02) raise if the sample rate is low, lower if the sample rate is high. The optimal value depends on the noise level of the IMU data and the dynamics of the motion being measured.
        :param theta_min: Minimum allowed elbow angle in degrees (default: 0.0)
        :param theta_max: Maximum allowed elbow angle in degrees (default: 140.0)
        """
        self.sample_rate = sample_rate
        self.madgwick_upper = Madgwick(sample_rate=sample_rate, beta=beta)
        self.madgwick_lower = Madgwick(sample_rate=sample_rate, beta=beta)
        self.q_upper = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion for upper IMU
        self.q_lower = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion for lower IMU
        self.gyr_bias_upper = [0.0, 0.0, 0.0]
        self.gyr_bias_lower = [0.0, 0.0, 0.0]
        self.zero = 0.0
        self.hinge_axis_samples = []
        self.theta_min = theta_min
        self.theta_max = theta_max

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
        
    def calculate_zeroing(self, imu_list: list):
        """
        Calculates the zeroing baseline for the elbow angle from a list containing the raw IMU data.
        
        :param imu_list: a list containing the raw IMU data, for the upper and lower arm, in the format [(acc_upper, gyr_upper), (acc_lower, gyr_lower)], where acc is the accelerometer data and gyr is the gyroscope data, over a 1 s window with the arm in a straight position.
        
        returns: the zeroing baseline for the elbow angle in degrees as a float
        """
        elbow_angles = []
        hinge_axis_samples = []
        for s in imu_list:
            s = np.asarray(s, dtype=float).reshape(-1)
            acc_u = s[0:3]
            gyr_u = s[3:6]
            acc_l = s[9:12]
            gyr_l = s[12:15]
            
            quat_u, quat_l = self.calculate_quarternions(acc_u, gyr_u, acc_l, gyr_l)
            q_rel = self._quat_mul(self._quat_conj(quat_u), quat_l)  # q_u^{-1} ⊗ q_l  (unit quats)

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
            
            elbow_angles.append(elbow_flex_deg)

        zeroing_baseline = np.mean(elbow_angles)
        self.zero = zeroing_baseline

        return self.zero

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
        na_u = np.linalg.norm(acc_upper)
        na_l = np.linalg.norm(acc_lower)
        if na_u < 1e-6 or na_l < 1e-6:
            return self.q_upper, self.q_lower  # Avoid division by zero, return previous quaternions
        
        acc_upper_normalized = acc_upper / na_u
        acc_lower_normalized = acc_lower / na_l

        # Update quaternions using Madgwick filter
        q_upper_new = self.madgwick_upper.updateIMU(self.q_upper, gyr=gyr_upper_rad, acc=acc_upper_normalized)
        q_lower_new = self.madgwick_lower.updateIMU(self.q_lower, gyr=gyr_lower_rad, acc=acc_lower_normalized)
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

        q_rel = self._quat_mul(self._quat_conj(quat_upper), quat_lower)  # q_u^{-1} ⊗ q_l  (unit quats)

        # Axis-angle from relative quaternion
        w = np.clip(q_rel[0], -1.0, 1.0)
        angle_rad = 2.0*np.arccos(w)
        sin_half = np.sqrt(max(1.0 - w*w, 1e-12))
        axis = q_rel[1:] / sin_half

        # Calculate signed hinge axis angle - This is casual, not completely simmilar to the previous code
        self.hinge_axis_samples.append(axis.copy())
        hinge_axis = np.mean(np.vstack(self.hinge_axis_samples), axis=0)
        hinge_axis /= np.linalg.norm(hinge_axis)

        # Signed hinge angle
        sign = np.sign(axis @ hinge_axis)
        elbow_flex_deg = np.degrees(angle_rad * sign)

        # Zero using first second (straight arm)
        elbow_flex_deg -= self.zero

        # Clip it to fit range of motion
        elbow_flex_deg = np.clip(elbow_flex_deg, self.theta_min, self.theta_max)
        return elbow_flex_deg

    def process_imu(self, acc_upper: np.ndarray, gyr_upper: np.ndarray, acc_lower: np.ndarray, gyr_lower: np.ndarray):
        """
        Convenience function to process the raw IMU data and directly get the elbow angle in degrees.
        
        :param acc_upper: Accelerometer data for the upper arm as a numpy array of shape (3,) containing the x, y, z accelerations in m/s^2
        :param gyr_upper: Gyroscope data for the upper arm as a numpy array of shape (3,) containing the x, y, z angular velocities in degrees/s
        :param acc_lower: Accelerometer data for the lower arm as a numpy array of shape (3,) containing the x, y, z accelerations in m/s^2
        :param gyr_lower: Gyroscope data for the lower arm as a numpy array of shape (3,) containing the x, y, z angular velocities in degrees/s
        
        returns: the elbow angle in degrees as a numpy array of shape (N,) where N is the number of samples, with positive values indicating flexion and negative values indicating extension.
        """
        quat_upper, quat_lower = self.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = self.calculate_elbow_angle(quat_upper, quat_lower)
        
        return elbow_angle
    
    @staticmethod
    def _quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_mul(q1, q2):
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)