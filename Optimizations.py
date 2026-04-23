import numpy as np

def _hard_clip(q, theta_min, theta_max):
    return max(min(q, theta_max), theta_min)

def _clip_tanh(q, theta_min, theta_max, sharpness=5.0):
    """
    Smoothly saturates q between theta_min and theta_max.
    
    sharpness:
        higher = sharper boundary (more like hard clip)
        lower = softer boundary
    """
    mid = 0.5 * (theta_max + theta_min)
    half_range = 0.5 * (theta_max - theta_min)

    # Normalize around center
    x = (q - mid) / half_range

    # Smooth saturation using tanh
    x_smooth = np.tanh(sharpness * x)

    # Map back to range
    return mid + half_range * x_smooth

def _soft_clip_tanh(q, theta_min, theta_max, margin=0.1, k=5.0):
    span = theta_max - theta_min
    margin = margin * span

    if q < theta_min + margin:
        return _clip_tanh(q, theta_min, theta_min + margin, sharpness=k)
    elif q > theta_max - margin:
        return _clip_tanh(q, theta_max - margin, theta_max, sharpness=k)
    else:
        return q

def _clip_sigmoid(q, theta_min, theta_max, k=5):
    span = theta_max - theta_min
    x = (q - theta_min) / span  # normalize to [0,1]

    s = 1 / (1 + np.exp(-k * (x - 0.5)))
    return theta_min + span * s

def _soft_clip_sigmoid(q, theta_min, theta_max, margin=0.1, k=5):
    span = theta_max - theta_min
    margin = margin * span

    if q < theta_min + margin:
        return _clip_sigmoid(q, theta_min, theta_min + margin, k=k)
    elif q > theta_max - margin:
        return _clip_sigmoid(q, theta_max - margin, theta_max, k=k)
    else:
        return q

def _clip_atan(q, theta_min, theta_max, k=3):
    mid = 0.5 * (theta_max + theta_min)
    half = 0.5 * (theta_max - theta_min)
    x = (q - mid) / half

    return mid + half * (2/np.pi) * np.arctan(k * x)

def _soft_clip_atan(q, theta_min, theta_max, margin=0.1, k=3):
    span = theta_max - theta_min
    margin = margin * span

    if q < theta_min + margin:
        return _clip_atan(q, theta_min, theta_min + margin, k=k)
    elif q > theta_max - margin:
        return _clip_atan(q, theta_max - margin, theta_max, k=k)
    else:
        return q

def _clip_smoothstep(q, theta_min, theta_max):
    x = (q - theta_min) / (theta_max - theta_min)
    x = np.clip(x, 0, 1)

    # cubic smoothstep
    x_smooth = x*x*(3 - 2*x)

    return theta_min + (theta_max - theta_min) * x_smooth

def _soft_clip_smoothstep(q, theta_min, theta_max, margin=0.1):
    span = theta_max - theta_min
    margin = margin * span

    if q < theta_min + margin:
        return _clip_smoothstep(q, theta_min, theta_min + margin)
    elif q > theta_max - margin:
        return _clip_smoothstep(q, theta_max - margin, theta_max)
    else:
        return q

def _boundary_scaling(q, theta_min, theta_max):
    span = theta_max - theta_min
    center = 0.5 * (theta_max + theta_min)
    dist = abs(q - center) / (span / 2)

    # 0 at center, 1 at boundary
    return 1 - dist**2

import numpy as np

def _boundary_scaling_1(q, activation, theta_min, theta_max, margin_ratio=0.1):
    span = theta_max - theta_min
    margin = margin_ratio * span

    if activation > 0:
        # moving toward theta_max
        dist = theta_max - q
    elif activation < 0:
        # moving toward theta_min
        dist = q - theta_min
    else:
        return 1.0

    if dist <= 0:
        return 0.0
    elif dist >= margin:
        return 1.0
    else:
        x = dist / margin
        # return x * x * (3 - 2 * x)  # cubic smoothstep
        return x**3 * (10 - 15*x + 6*x**2) # quintic smoothstep

# ==============================================================================================================================

MARGIN_RATIO = 0.2

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
    scale = _boundary_scaling_1(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)

    delta_q = k * activation * t * scale
    optimized_angle = q + delta_q
    
    # optimized_angle = _hard_clip(optimized_angle, theta_min, theta_max)
    # optimized_angle = _soft_clip_tanh(optimized_angle, theta_min, theta_max, margin=0.1, k=5.0)
    # optimized_angle = _soft_clip_sigmoid(optimized_angle, theta_min, theta_max, k=5)
    # optimized_angle = _soft_clip_atan(optimized_angle, theta_min, theta_max, margin=0.1, k=3)
    # optimized_angle = _soft_clip_smoothstep(optimized_angle, theta_min, theta_max)
    
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
    
    # optimized_angle = _hard_clip(optimized_angle, theta_min, theta_max)
    # optimized_angle = _soft_clip_tanh(optimized_angle, theta_min, theta_max, sharpness=5.0)
    # optimized_angle = _soft_clip_sigmoid(optimized_angle, theta_min, theta_max, k=5)
    # optimized_angle = _soft_clip_atan(optimized_angle, theta_min, theta_max, k=3)
    # optimized_angle = _soft_clip_smoothstep(optimized_angle, theta_min, theta_max)
    
    return optimized_angle

def optimize_3(k, activation, t, q, theta_min, theta_max, deadband=0.1):
    if abs(activation) < deadband:
        a_eff = 0
    else:
        a_eff = activation

    dq = k * a_eff * t
    q_next = q + dq

    # q_next = _hard_clip(q_next, theta_min, theta_max)
    # q_next = _soft_clip_tanh(q_next, theta_min, theta_max, sharpness=5.0)
    # q_next = _soft_clip_sigmoid(q_next, theta_min, theta_max, k=5)
    # q_next = _soft_clip_atan(q_next, theta_min, theta_max, k=3)
    # q_next = _soft_clip_smoothstep(q_next, theta_min, theta_max)

    return q_next

def optimize_4(k, activation, t, q, delta_q_prev, theta_min, theta_max, alpha=0.5):
    scale = _boundary_scaling_1(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    
    delta_q_raw = k * activation * t * scale
    delta_q = alpha * delta_q_raw + (1-alpha) * delta_q_prev
    
    optimized_angle = q + delta_q
    
    # optimized_angle = _hard_clip(optimized_angle, theta_min, theta_max)
    # optimized_angle = _soft_clip_tanh(optimized_angle, theta_min, theta_max, sharpness=5.0)
    # optimized_angle = _soft_clip_sigmoid(optimized_angle, theta_min, theta_max, k=5)
    # optimized_angle = _soft_clip_atan(optimized_angle, theta_min, theta_max, k=3)
    # optimized_angle = _soft_clip_smoothstep(optimized_angle, theta_min, theta_max)

    return optimized_angle, delta_q

def optimize_5_pd(activation, velocity, t, q, theta_min, theta_max, v_max, k, b=0.5,):
    scale = _boundary_scaling_1(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    velocity = b * velocity + k * activation * scale
    velocity = np.clip(velocity, -v_max, v_max)
    
    q_next = q + velocity * t
    
    # q_next = _hard_clip(q_next, theta_min, theta_max)
    # q_next = _soft_clip_tanh(q_next, theta_min, theta_max, sharpness=5.0)
    # q_next = _soft_clip_sigmoid(q_next, theta_min, theta_max, k=5)
    # q_next = _soft_clip_atan(q_next, theta_min, theta_max, k=3)
    # q_next = _soft_clip_smoothstep(q_next, theta_min, theta_max)

    return q_next, velocity

def optimizer_6(activation, velocity, t, q, theta_min, theta_max, v_max=np.pi, b = 6.0, k = None):
    # Smoothen acceleration
    k = b * np.pi if k is None else k
    scale = _boundary_scaling_1(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    acc = k * activation * scale - b * velocity

    # Update velocity and position
    velocity += acc * t
    velocity = np.clip(velocity, -v_max, v_max)
    q_next = q + velocity * t * scale

    # q_next = _hard_clip(q_next, theta_min, theta_max)
    # q_next = _soft_clip_tanh(q_next, theta_min, theta_max, sharpness=5.0)
    # q_next = _soft_clip_sigmoid(q_next, theta_min, theta_max, k=5)
    # q_next = _soft_clip_atan(q_next, theta_min, theta_max, k=3)
    # q_next = _soft_clip_smoothstep(q_next, theta_min, theta_max)

    # # boundary handling (soft)
    # if q_next < theta_min:
    #     q_next = theta_min
    #     if velocity < 0: velocity = 0
    # elif q_next > theta_max:
    #     q_next = theta_max
    #     if velocity > 0: velocity = 0

    # If i want to smooth velocity:
    # # --- Smooth velocity damping near boundaries ---
    # margin = 0.1 * (theta_max - theta_min)

    # dist_min = q_next - theta_min
    # dist_max = theta_max - q_next
    # dist = min(dist_min, dist_max)

    # # Smooth scaling factor (0 → at boundary, 1 → center)
    # scale = np.clip(dist / margin, 0.0, 1.0)

    # # Apply damping
    # velocity *= scale

    return q_next, velocity, acc

def EMG_Optimizer(a, d_a, v, kn, kd, b, q, THETA_MIN, THETA_MAX, v_max, t):
   scale = _boundary_scaling_1(q, a, THETA_MIN, THETA_MAX, margin_ratio=MARGIN_RATIO)
   
   # Calculate desired acceleration
   acc = (kn * a + kd * d_a - b * v) * scale

   # Update velocity and position
   v += t * acc
   v = np.clip(v, -v_max, v_max)
   q_next = q + t * v * scale

   return q_next, v, acc

def EMG_IMU_optimizer(a, d_a, v, omega, kn, kd, kp, b, q, imu_q, theta_min, theta_max, v_max, t):
    scale = _boundary_scaling_1(q, a, theta_min, theta_max, margin_ratio=MARGIN_RATIO)

    # Calculate desired acceleration
    acc = (kn * a + kd * d_a - b * (v - omega) - kp * (q - imu_q)) * scale

    # Update velocity and position
    v += t * acc
    v = np.clip(v, -v_max, v_max)
    q_next = q + t * v * scale

    return q_next, v, acc

def EMG_IMU_optimizer_2(a, d_a, omega, kn, kd, imu_q, theta_min, theta_max, v_max, t):
    # Calculate desired velocity
    v = omega + kn * a + kd * d_a
    v = np.clip(v, -v_max, v_max)

    # Update position
    scale = _boundary_scaling_1(imu_q, a, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    q_next = imu_q + t * v * scale

    return q_next, v