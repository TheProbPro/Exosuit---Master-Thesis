def optimize_1(k, activation, t, q, theta_min, theta_max):
    """
    Optimizes movement based on EMG signal
    Parameters:
    k: maximum angular velocity (degrees per second)
    activation: muscle activation level (-1 to 1)
    t: time between updates (seconds)
    q: current angle (degrees)
    theta_min: minimum angle of the movement
    theta_max: maximum angle of the movement
    Returns:
    optimized_angle: the optimized angle for the movement
    """
    delta_q = k * activation * t
    optimized_angle = q + delta_q
    optimized_angle = max(min(optimized_angle, theta_max), theta_min)
    return optimized_angle


# TODO: Maybe this needs to be changed to use the difference between the current and previous angle to decide the direction of motion.
def optimize_2(k, activation, t, q, theta_min, theta_max):
    """
    Optimizes movement based on EMG signal
    Parameters:
    k: maximum angular velocity (degrees per second)
    activation: muscle activation level (-1 to 1)
    t: time between updates (seconds)
    q: current angle (degrees)
    theta_min: minimum angle of the movement
    theta_max: maximum angle of the movement
    Returns:
    optimized_angle: the optimized angle for the movement
    """
    w = 0
    if activation > 0:
        w = (theta_max - q) / theta_max
    elif activation < 0:
        w = q / theta_max
    # w = (theta_max - q) / theta_max

    delta_q = k * activation * t * w
    optimized_angle = q + delta_q
    optimized_angle = max(min(optimized_angle, theta_max), theta_min)
    return optimized_angle

def optimize_3(k, activation, t, q, theta_min, theta_max, deadband=0.1):
    if abs(activation) < deadband:
        a_eff = 0
    else:
        a_eff = activation

    dq = k * a_eff * t
    q_next = q + dq

    q_next = max(0, min(140, q_next))
    return q_next

def optimize_4(k, activation, t, q, delta_q_prev, theta_min, theta_max, alpha=0.5):
    delta_q_raw = k * activation * t
    delta_q = alpha * delta_q_raw + (1-alpha) * delta_q_prev
    optimized_angle = q + delta_q
    optimized_angle = max(min(optimized_angle, theta_max), theta_min)
    return optimized_angle, delta_q_prev

def optimize_5_pd(activation, velocity, t, q, theta_min, theta_max, k, b=0.5,):
    velocity = b * velocity + k * activation
    q_next = q + velocity * t
    q_next = max(theta_min, min(theta_max, q_next))
    return q_next

