from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6
import AdaptiveEmbodiedControlSystems.LSTM as LSTM

# Control imports

# motor imports

# normal imports
import time
import queue
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import signal
import matplotlib.pyplot as plt

# ======================= Global Parameters =======================
FS           = 2000          # EMG 采样率 (Hz)
EMG_DT       = 1.0 / FS
USER_NAME    = 'VictorBNielsen'
LSTM_PATH    = "Outputs/models/LSTM/Windowed_LSTM.pth"

# EMG 优化器参数（与原 EMG 脚本保持一致）
EMG_B        = 4.0
EMG_K        = np.pi * 10.0 * 2

SINE_CENTER_DEG  = 0.0          # 归中位置（对应 THETA 中点）
# ── 关节范围（EMG 和控制器共享）─────────────────────────────
THETA_MIN    = np.deg2rad(0)    # 0 rad
THETA_MAX    = np.deg2rad(140)  # ~2.44 rad
THETA_RANGE  = THETA_MAX - THETA_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_event = threading.Event()
plota = []
plotq = []
plot_pred = []
plotdq = []
pttest = []

window_lock = threading.Lock()

#=================================================================

def emg_thread_fn(model, qd_queue: deque):
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

    # optimizer_6 状态
    emg_v             = 0.0
    optimized_angle   = float(np.deg2rad(SINE_CENTER_DEG))  # 从中心位置起步
    sample_counter = 0

    # buffer = deque(np.zeros(2000), maxlen=2000)  # 用于存储原始 net_a，供后续分析和调试

    # plt.ion()
    # fig, ax = plt.subplots(figsize=(10, 5))
    # line, = ax.plot(np.arange(len(buffer)), np.zeros(len(buffer)), lw=1)
    # ax.set_ylim(0, 2.4)
    # plt.tight_layout()
    # plt.show()
    
    # 启动传感器
    emg = DelsysEMG(channel_range=(0, 1))
    emg.start()
    # time.sleep(1.0)
    print("[EMG] 线程启动，开始采集...")

    while not stop_event.is_set():
        st = time.time()
        reading    = emg.read()

        sample_counter += 1
        
        # 带通滤波
        filtered_bicep  = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        # 滑动 RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get_nowait()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get_nowait()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS  = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms  = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        # 激活度 → 净激活
        activation   = interpreter.compute_activation(
            [filtered_bicep_rms, filtered_tricep_rms]
        )
        net_a        = activation[0] - activation[1]
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        # optimizer_6 → 平滑角度
        optimized_angle, emg_v, _ = optimizer_6(
            filtered_net_a, emg_v, EMG_DT,
            optimized_angle, THETA_MIN, THETA_MAX,
            np.pi, EMG_B, EMG_K
        )
        plota.append(filtered_net_a)
        plotq.append(optimized_angle)

        with window_lock:
            qd_queue.append(optimized_angle)

        if len(qd_queue) < qd_queue.maxlen:
            continue

        if len(qd_queue) == qd_queue.maxlen and sample_counter % 10 == 0:
            with torch.inference_mode():
                input_tensor = torch.as_tensor(
                    qd_queue,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).unsqueeze(-1)
                lstm_output     = model(input_tensor)
                predicted_angle = float(lstm_output.detach().cpu().item())
                plot_pred.append(predicted_angle)
        pttest.append(time.time() - st)

    

    # plt.ioff()
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
    qd_queue = deque(maxlen=100)

    # initialize prediction model
    # LSTM 模型
    model = LSTM.LSTMModel(
        input_size=1, hidden_size=64,
        output_size=1, num_layers=1, batch_first=True
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    # plot_pred = []

    # Start EMG thread
    emg_thread = threading.Thread(target=emg_thread_fn, args=(model, qd_queue,))
    emg_thread.start()

    while len(qd_queue) < qd_queue.maxlen:
        time.sleep(0.01)  # 等待队列填满

    start = time.time()
    while time.time() - start < 10:
        loop_start = time.perf_counter()

        # if len(qd_queue) == qd_queue.maxlen:
        #     with window_lock:
        #         window_copy = np.array(qd_queue, dtype=np.float32)
        #     prediction = predict(model, window_copy)
        #     plot_pred.append(prediction)

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, 1/200 - elapsed))

    stop_event.set()
    emg_thread.join()

    # print(f"Average LSTM prediction time: {np.mean(ptitime)*1000:.2f}ms")
    print(f"Processing time per loop: mean={np.mean(pttest)*1000:.2f}ms")

    # create time vector
    print(f"length of plot_pred: {len(plot_pred)}, length of plotq: {len(plotq)}")
    t_vec = np.arange(len(plot_pred)) * 1/200  # 200Hz采样率对应的时间步长
    t_vec2 = np.arange(len(plotq)) * 1/2000 # 2000Hz采样率对应的时间步长

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_vec, plot_pred, label='Predicted Angle (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.subplot(2, 1, 2)
    plt.plot(t_vec2, plotq, label='Optimizer Output Angle (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
