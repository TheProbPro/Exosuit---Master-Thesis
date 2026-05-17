import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    
    'font.size': 10,          # default text size
    'axes.titlesize': 14,     # title
    'axes.labelsize': 12,     # x and y labels
    'xtick.labelsize': 10,    # x tick labels
    'ytick.labelsize': 10,    # y tick labels
    'legend.fontsize': 10,    
    'figure.titlesize': 16
})

import ESN  # 你的ESN模块

TRAIN_CSV = "Outputs/RecordedEMG/Optim2/TrainLSTM.csv"
TEST_CSV = "Outputs/RecordedEMG/Optim2/TestLSTM.csv"
#COL = 'Processed EMG'  # 'Position'
COL = 'emg_pos'

# PREDICT_X = 40 #20ms
# PREDICT_X = 60 #30ms
PREDICT_X = 80 #40ms

TRAIN = True

Model_Save_Path = "Outputs/models/ESN/Windowed_ESN_80.pth"

class CSVWindowedDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        super().__init__()

        # Read data column from CSV file
        df = pd.read_csv(csv_file)
        data = df[COL].values.astype(np.float32)
        X_list = []
        y_list = []
        
        for i in range(len(data) - seq_len - PREDICT_X):
            window = data[i : i + seq_len]        # (seq_len,)
            target = data[i + seq_len + PREDICT_X]            # scalar
            X_list.append(window[:, None])  # (seq_len, 1)
            y_list.append([target])          # (1,)
        self.X = torch.from_numpy(np.stack(X_list, axis=0))  # (N, seq_len, 1), float32
        self.y = torch.from_numpy(np.stack(y_list, axis=0))  # (N, 1), float32

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CSVContinuousDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()

        # Read data column from CSV file
        df = pd.read_csv(csv_file)
        data = df[COL].values.astype(np.float32)
        self.signal = torch.tensor(data[:, None])  # (T, 1)

    def __len__(self):
        return self.signal.shape[0]
    
    def __getitem__(self, idx):
        return self.signal[idx]

# 添加输出激活函数的ESN包装器
class ESNWithActivation(nn.Module):
    def __init__(self, esn_model, activation='softplus'):
        super(ESNWithActivation, self).__init__()
        self.esn = esn_model
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x, state=None):
        outputs, final_state = self.esn(x, state)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs, final_state

class WindowedESNWithActivation(nn.Module):
    def __init__(self, esn_model, activation='softplus'):
        super(WindowedESNWithActivation, self).__init__()
        self.esn = esn_model
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x, state=None):
        # WindowedESN只返回一个值，不是两个
        output = self.esn(x, state)
        if self.activation is not None:
            output = self.activation(output)
        return output

def train_windowed_esn():
    # hyperparameters
    # seq_length = 25
    seq_length = 100
    reservoir_size = 100
    spectral_radius = 0.9
    leaking_rate = 0.7
    connectivity = 0.1
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_loss_array = np.array([])
    total_loss_array = np.array([])

    # data
    dataset = CSVWindowedDataset(TRAIN_CSV, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 使用带激活函数的WindowedESN
    base_model = ESN.WindowedESN(
        input_size=1, 
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=spectral_radius,
        leaking_rate=leaking_rate,
        connectivity=connectivity
    )
    model = WindowedESNWithActivation(base_model, activation='softplus').to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for xb, yb in loader:
            # xb: (batch, seq_len, 1)
            # yb: (batch, 1)
            pred = model(xb.to(device))
            loss = criterion(pred, yb.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
        total_loss_array = np.append(total_loss_array, total_loss)

        avg_loss = total_loss / len(dataset)
        average_loss_array = np.append(average_loss_array, avg_loss)
        print(f"epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

    return model, average_loss_array, total_loss_array

def train_continuous_esn():
    # Hyperparameters
    reservoir_size = 100
    spectral_radius = 0.9
    leaking_rate = 0.7
    connectivity = 0.1
    learning_rate = 0.01
    num_epochs = 25
    seq_length = 25  # 用于连续训练的序列长度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_loss_array = np.array([])
    total_loss_array = np.array([])

    # Data
    dataset = CSVContinuousDataset(TRAIN_CSV)
    full_signal = dataset.signal.to(device)
    T = full_signal.shape[0]

    # 使用带激活函数的ESN
    base_model = ESN.ESN(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=spectral_radius,
        leaking_rate=leaking_rate,
        connectivity=connectivity
    )
    model = ESNWithActivation(base_model, activation='softplus').to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        count = 0
        
        # 每个epoch开始时重置状态
        state = None

        # 使用滑动窗口的方式进行训练
        for t in range(0, T - seq_length - 1, 1):  # 步长为1，重叠窗口
            # 获取当前窗口
            window_start = t
            window_end = t + seq_length
            target_idx = t + seq_length
            
            # 创建输入窗口和目标
            x_window = full_signal[window_start:window_end].unsqueeze(0)  # (1, seq_len, 1)
            y_target = full_signal[target_idx].view(1, 1)  # (1, 1)
            
            # 前向传播
            all_outputs, state = model(x_window, state)
            
            # 只使用最后一个时间步的输出作为预测
            pred = all_outputs[:, -1, :]
            
            loss = criterion(pred, y_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            count += 1
            
            # 定期detach状态以防止梯度爆炸
            if state is not None and count % 50 == 0:
                state = state.detach()

        avg_loss = running_loss / count if count > 0 else 0
        total_loss_array = np.append(total_loss_array, running_loss)
        average_loss_array = np.append(average_loss_array, avg_loss)
        
        print(f"epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.6f}")

    return model, average_loss_array, total_loss_array

@torch.no_grad()
def evaluate_windowed_esn(model, seq_len, device="cpu", total_points=1000):
    """评估窗口式ESN模型"""
    model = model.to(device)
    model.eval()

    df = pd.read_csv(TEST_CSV)
    data = df[COL].values.astype(np.float32)
    signal = torch.tensor(data[:, None], device=device)
    t_axis = torch.arange(signal.shape[0], device=device)
    T = signal.shape[0]

    X_list = []
    for i in range(T - seq_len - PREDICT_X):
        window = signal[i : i+seq_len]
        X_list.append(window.unsqueeze(0))

    if len(X_list) == 0:
        raise ValueError("total_points too small vs seq_len during evaluation")

    X_batch = torch.cat(X_list, dim=0)
    # preds = model(X_batch)
    batch_size = 512  # or 256 if needed

    preds_list = []
    timing = []
    for i in range(0, X_batch.shape[0], batch_size):
        t = time.time()
        xb = X_batch[i:i+batch_size]
        preds = model(xb)
        preds_list.append(preds.cpu())
        timing.append(time.time() - t)

    print(f"Avg batch inference time: {np.mean(timing):.4f} seconds per batch of size {batch_size}")
    preds = torch.cat(preds_list, dim=0)
    y_pred = preds.squeeze(-1).cpu().numpy()

    target_indices = torch.arange(seq_len + PREDICT_X, seq_len + PREDICT_X + len(y_pred))
    y_true = signal[target_indices, 0].cpu().numpy()
    t_pred = t_axis[target_indices].cpu().numpy()

    return t_pred, y_true, y_pred

@torch.no_grad()
def evaluate_continuous_esn(model, seq_len=25, device="cpu", total_points=1000):
    """评估连续ESN模型"""
    model = model.to(device)
    model.eval()

    df = pd.read_csv(TEST_CSV)
    data = df[COL].values.astype(np.float32)
    signal = torch.tensor(data[:, None], device=device)
    t_axis = torch.arange(signal.shape[0], device=device)
    T = signal.shape[0]

    preds_list = []
    t_idx_list = []
    state = None

    # 使用滑动窗口的方式进行预测
    for t in range(T - seq_len - 1):
        window = signal[t:t+seq_len].unsqueeze(0)  # (1, seq_len, 1)
        target_idx = t + seq_len
        
        outputs, state = model(window, state)
        pred = outputs[:, -1, :]  # 最后一个时间步的输出
        
        preds_list.append(pred.item())
        t_idx_list.append(target_idx)

    t_idx_tensor = torch.tensor(t_idx_list, device=device)
    t_pred = t_axis[t_idx_tensor].cpu().numpy()
    y_true = signal[t_idx_tensor, 0].cpu().numpy()
    y_pred = np.array(preds_list)

    return t_pred, y_true, y_pred

def plot_predictions(t_pred, y_true, y_pred, title):
    plt.figure(figsize=(7,3))
    plt.plot(t_pred, y_true, label="Ground truth", linewidth=2)
    plt.plot(t_pred, y_pred, label="ESN prediction", linestyle='--')
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.title(title, fontsize=12)
    plt.xlim([t_pred[0], t_pred[-1]])
    plt.legend(fontsize=10, loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))

    mse = np.mean((y_true - y_pred) ** 2)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # avoid division by zero
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denom if denom != 0 else np.nan

    return mae, mse, rmse, r2

if __name__ == "__main__":
    if TRAIN:
        print("Training Windowed ESN...")
        WindowedESN, Windowed_avg_loss, Windowed_total_loss = train_windowed_esn()
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(Windowed_avg_loss, label='Average Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Loss over Epochs (Windowed ESN)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(Windowed_total_loss, label='Total Loss per Epoch', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Loss over Epochs (Windowed ESN)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # 加载预训练的Windowed ESN模型
        basemodel = ESN.WindowedESN(
            input_size=1, 
            reservoir_size=100,
            output_size=1,
            spectral_radius=0.9,
            leaking_rate=0.7,
            connectivity=0.1
        )
        WindowedESN = WindowedESNWithActivation(basemodel, activation='softplus')
        WindowedESN.load_state_dict(torch.load(Model_Save_Path, map_location=torch.device('cpu')))

    # print("Training Continuous ESN...")
    # ContinuousESN, Continuous_avg_loss, Continuous_total_loss = train_continuous_esn()
    
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(Continuous_avg_loss, label='Average Loss per Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')
    # plt.title('Average Loss over Epochs (Continuous ESN)')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(Continuous_total_loss, label='Total Loss per Epoch', color='orange')
    # plt.xlabel('Epoch')
    # plt.ylabel('Total Loss')
    # plt.title('Total Loss over Epochs (Continuous ESN)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # 评估和绘图
    # seq_length_for_eval = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length_for_eval = 100
    print("Evaluating Windowed ESN...")
    t_pred_w, y_true_w, y_pred_w = evaluate_windowed_esn(
        model=WindowedESN,
        seq_len=seq_length_for_eval,
        # device="cpu",
        device=device,
        total_points=1000
    )
    plot_predictions(
        t_pred_w,
        y_true_w,
        y_pred_w,
        title="Windowed ESN Prediction vs True Signal"
    )
    # metrics
    mae_w, mse_w, rmse_w, r2_w = compute_metrics(y_true_w, y_pred_w)
    print("\n--- Windowed ESN Metrics ---")
    print(f"MAE  : {mae_w:.6f}")
    print(f"MSE  : {mse_w:.6f}")
    print(f"RMSE : {rmse_w:.6f}")
    print(f"R^2  : {r2_w:.6f}")

    error = y_true_w - y_pred_w

    plt.figure(figsize=(7,3))
    plt.plot(t_pred_w, error)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Prediction Error Over Time")
    plt.grid()
    plt.show()

    abs_error = np.abs(y_true_w - y_pred_w)

    plt.figure(figsize=(7,3))
    plt.plot(t_pred_w, abs_error)
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error Over Time")
    plt.grid()
    plt.show()

    low, high = np.percentile(error, [1, 99])  # keep central 98%

    plt.figure(figsize=(7,3))
    plt.hist(y_true_w - y_pred_w, bins=50)
    plt.xlabel("Error")
    plt.xlim([low, high])
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Outputs/PredictionResults/ESNErrorDist.png", dpi=300)
    # plt.title("Error Distribution")
    plt.show()

    cum_mae = np.cumsum(np.abs(y_true_w - y_pred_w)) / np.arange(1, len(y_true_w)+1)

    plt.figure(figsize=(7,3))
    plt.plot(cum_mae)
    plt.xlabel("Samples")
    plt.xlim([0, len(cum_mae)])
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig("Outputs/PredictionResults/ESNCumError.png", dpi=300)
    # plt.title("Cumulative MAE")
    plt.show()

    # Save the windowed ESN model and make sure the save path exists
    if TRAIN:
        save_dir = os.path.dirname(Model_Save_Path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(WindowedESN.state_dict(), Model_Save_Path)

    # print("Evaluating Continuous ESN...")
    # t_pred_c, y_true_c, y_pred_c = evaluate_continuous_esn(
    #     model=ContinuousESN,
    #     seq_len=seq_length_for_eval,
    #     device="cpu",
    #     total_points=1000
    # )
    # plot_predictions(
    #     t_pred_c,
    #     y_true_c,
    #     y_pred_c,
    #     title="Continuous ESN Prediction vs True Signal"
    # )