# EMG processing imports
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6
import AdaptiveEmbodiedControlSystems.LSTM as LSTM
from Motors.DynamixelHardwareInterface import Motors

# Control imports

# motor imports

# normal imports
import time
from collections import deque
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import signal

# ======================= Global Parameters =======================
FS = 2000  # Sampling frequency for EMG
dt = 1 / FS  # Time step
THETA_MIN = np.deg2rad(0)  # Minimum angle in radians
THETA_MAX = np.deg2rad(140)  # Maximum angle in radians
THETA_RANGE = THETA_MAX - THETA_MIN  # Range of motion in radians
USER_NAME = 'VictorBNielsen'  # User name for EMG interpretation
TORQUE_MIN = -4.1  # Minimum torque for motors
TORQUE_MAX = 4.1  # Maximum torque for motors
MAX_VEL = np.pi

EMG_B        = 4.0
EMG_K        = np.pi * 10.0 * 2

LSTM_PATH = "Outputs/models/LSTM/Windowed_LSTM.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_event = threading.Event()

#=================================================================

def emg_thread_fn(qd_queue: queue.Queue):
    """
    采集肌电 → 滤波 → 激活度 → optimizer_6 → LSTM → predicted_angle
    结果写入 qd_queue，控制器取最新值。

    qd_queue 元素格式：(theta_d_rad, timestamp)
    """
    # 滤波器
    filter_bicep  = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(FS, lp_cutoff=2, order=2)

    # 激活度解释器
    interpreter = PMC(
        theta_min=THETA_MIN, theta_max=THETA_MAX,
        user_name=USER_NAME, BicepEMG=True, TricepEMG=True
    )

    # RMS 滑动窗口
    Bicep_RMS_queue  = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # LSTM 模型
    model = LSTM.LSTMModel(
        input_size=1, hidden_size=64,
        output_size=1, num_layers=1, batch_first=True
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    # optimizer_6 状态
    window = deque(maxlen=100)  # 50 samples at 2000Hz = 25ms window
    emg_v             = 0.0
    optimized_angle   = float(np.deg2rad(0))  # 从中心位置起步
    sample_counter = 0
    net_a_prev = 0.0

    # 启动传感器
    emg = DelsysEMG(channel_range=(0, 1))
    emg.start()
    # time.sleep(1.0)
    print("[EMG] 线程启动，开始采集...")

    while not stop_event.is_set():
        reading    = emg.read()
        sample_counter += 1

        # 带通滤波
        filtered_bicep  = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        # 滑动 RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get_nowait()
        Bicep_RMS_queue.put_nowait(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get_nowait()
        Tricep_RMS_queue.put_nowait(filtered_tricep)

        Bicep_RMS  = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms  = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        # 激活度 → 净激活
        activation   = interpreter.compute_activation(
            [filtered_bicep_rms, filtered_tricep_rms]
        )
        # Standard:
        # net_a = activation[0] - activation[1]

        # Normalize activation[0] (bicep activation) to [-1,1]
        net_a = 2 * activation[0] - 1.0
        # Alternatively use temporal differnece, Try with both standard and bicep.
        # net_a = net_a - net_a_prev
        # net_a_prev = net_a

        # net_a        = activation[0] - activation[1]
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        # optimizer_6 → 平滑角度
        optimized_angle, emg_v, _ = optimizer_6(
            filtered_net_a, emg_v, dt,
            optimized_angle, THETA_MIN, THETA_MAX,
            np.pi, EMG_B, EMG_K
        )

        window.append(optimized_angle)
        if len(window) < window.maxlen:
            continue

        if len(window) == window.maxlen and sample_counter % 10 == 0:
            with torch.inference_mode():
                input_tensor = torch.as_tensor(
                    window,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).unsqueeze(-1)
                lstm_output     = model(input_tensor)
                predicted_angle = float(lstm_output.detach().cpu().item())

            try:
                qd_queue.put_nowait((predicted_angle))
            except queue.Full:
                qd_queue.get_nowait()
                qd_queue.put_nowait((predicted_angle))

    # 清理
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    print("[EMG] 线程已停止。")

# ==========================================================================================================================
# =========================================== Handle cntrl+c stop event ====================================================
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

# ==========================================================================================================================
# ======================================================= Main function ====================================================

if __name__ == "__main__":
    # Desired position queue
    qd_queue = queue.Queue(maxsize=3) # or 5

    #TODO:Define parameters here if needed otherwise do it in global parameters section

    # TODO: Initialize motors here
    motor = Motors(port="COM4")
    motor.enable_torque()
    time.sleep(1.0)  # Allow some time for motor initialization

    emg_thread = threading.Thread(target=emg_thread_fn, args=(qd_queue,))
    emg_thread.start()

    # TODO: Here you would add the code to read from qd_queue and send commands to the exoskeleton motors
    while not stop_event.is_set():
        pass

    # If we dont want to manually stop it we just have to call this after while loop
    # stop_event.set()here

    emg_thread.join()


