import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP冲突

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# 导入模型
from ESN import ESN, WindowedESN

def generate_test_sine(total_points=1000, noise_std=0.0, device="cpu"):
    """
    Create a clean (or slightly noisy) sine wave to test the trained model.
    """
    t_axis = torch.linspace(0, 20*np.pi, steps=total_points)
    clean = torch.sin(t_axis)
    if noise_std > 0:
        clean = clean + noise_std * torch.randn_like(clean)

    signal = clean.unsqueeze(-1).to(device)
    return signal, t_axis

class SineDataset(Dataset):
    '''A dataset of sine wave sequences for training.'''
    def __init__(self, total_points=5000, seq_len=50):
        super().__init__()
        self.seq_len = seq_len

        t = torch.linspace(0, 20*np.pi, steps=total_points)
        full_signal = torch.sin(t)

        X_list = []
        y_list = []
        for i in range(total_points - seq_len - 1):
            window = full_signal[i : i+seq_len]
            target = full_signal[i+seq_len]
            X_list.append(window.unsqueeze(-1))
            y_list.append(target.unsqueeze(-1))

        self.X = torch.stack(X_list, dim=0)
        self.y = torch.stack(y_list, dim=0)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ContinuousSineDataset(Dataset):
    """
    A continuous sine-like signal for streaming training.
    """
    def __init__(self, total_points=5000):
        super().__init__()

        t = torch.linspace(0, 20*np.pi, steps=total_points)
        signal = torch.sin(t)

        noise = 0.05 * torch.randn_like(signal)
        signal = signal + noise

        self.signal = signal.unsqueeze(-1)

    def __len__(self):
        return self.signal.size(0)

    def __getitem__(self, idx):
        return self.signal[idx]

# ==================== ESN 训练函数 ====================
def train_windowed_esn():
    # Hyperparameters - 与LSTM保持一致
    seq_length = 25
    reservoir_size = 64
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10  # 减少epochs快速测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    average_loss_array = np.array([])
    total_loss_array = np.array([])

    # Data
    dataset = SineDataset(total_points=2000, seq_len=seq_length)  # 减少数据量
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = WindowedESN(
        input_size=1, 
        reservoir_size=reservoir_size, 
        output_size=1,
        spectral_radius=0.95,
        leaking_rate=0.7,
        connectivity=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb.to(device))
            loss = criterion(pred, yb.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
        
        total_loss_array = np.append(total_loss_array, total_loss)
        avg_loss = total_loss / len(dataset)
        average_loss_array = np.append(average_loss_array, avg_loss)
        print(f"Windowed ESN epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

    return model, average_loss_array, total_loss_array

def train_continuous_esn():
    # Hyperparameters - 与LSTM保持一致
    reservoir_size = 64
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 10  # 减少epochs快速测试
    tbptt_steps = 50
    total_points = 2000  # 减少数据量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    average_loss_array = np.array([])
    total_loss_array = np.array([])

    # Data
    dataset = ContinuousSineDataset(total_points=total_points)
    full_signal = dataset.signal.to(device)
    T = full_signal.shape[0]

    # Model
    model = ESN(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=0.95,
        leaking_rate=0.7,
        connectivity=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        reservoir_state = None
        running_loss = 0.0
        step_in_chunk = 0
        chunk_losses = []

        for t in range(T - 1):
            x_t = full_signal[t].view(1, 1, 1)
            y_target = full_signal[t+1].view(1, 1, 1)

            preds, reservoir_state = model(x_t, reservoir_state)
            loss_t = criterion(preds[:, -1, :], y_target)
            chunk_losses.append(loss_t)
            step_in_chunk += 1

            if step_in_chunk == tbptt_steps:
                loss_chunk = torch.stack(chunk_losses).mean()
                optimizer.zero_grad()
                loss_chunk.backward()
                optimizer.step()
                reservoir_state = reservoir_state.detach()
                running_loss += loss_chunk.item()
                chunk_losses = []
                step_in_chunk = 0

        if step_in_chunk > 0:
            loss_chunk = torch.stack(chunk_losses).mean()
            optimizer.zero_grad()
            loss_chunk.backward()
            optimizer.step()
            reservoir_state = reservoir_state.detach()
            running_loss += loss_chunk.item()

        total_loss_array = np.append(total_loss_array, running_loss)
        denom = max(1, (T // tbptt_steps))
        avg_loss = running_loss / denom
        average_loss_array = np.append(average_loss_array, avg_loss)
        print(f"Continuous ESN epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.6f}")

    return model, average_loss_array, total_loss_array

# ==================== 评估函数 ====================
@torch.no_grad()
def evaluate_windowed_esn(model, seq_len, device="cpu", total_points=1000):
    model = model.to(device)
    model.eval()

    signal, t_axis = generate_test_sine(total_points=total_points, noise_std=0.0, device=device)
    T = signal.shape[0]

    X_list = []
    for i in range(T - seq_len - 1):
        window = signal[i : i+seq_len]
        X_list.append(window.unsqueeze(0))

    X_batch = torch.cat(X_list, dim=0)
    preds = model(X_batch)
    
    y_pred = preds.squeeze(-1).cpu().numpy()
    target_indices = torch.arange(seq_len, seq_len + len(y_pred))
    y_true = signal[target_indices, 0].cpu().numpy()
    t_pred = t_axis[target_indices].cpu().numpy()

    return t_pred, y_true, y_pred

@torch.no_grad()
def evaluate_continuous_esn(model, device="cpu", total_points=1000):
    model = model.to(device)
    model.eval()

    signal, t_axis = generate_test_sine(total_points=total_points, noise_std=0.0, device=device)
    T = signal.shape[0]

    reservoir_state = None
    preds_list = []
    t_idx_list = []

    for t in range(T - 1):
        x_t = signal[t].view(1, 1, -1)
        pred, reservoir_state = model(x_t, reservoir_state)
        preds_list.append(pred[:, -1, :].item())
        t_idx_list.append(t + 1)
        reservoir_state = reservoir_state.detach()

    t_idx_tensor = torch.tensor(t_idx_list, device=device)
    t_pred = t_axis[t_idx_tensor].cpu().numpy()
    y_true = signal[t_idx_tensor, 0].cpu().numpy()
    y_pred = np.array(preds_list)

    return t_pred, y_true, y_pred

def plot_predictions(t_pred, y_true, y_pred, title):
    plt.figure(figsize=(10,4))
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

# ==================== 主函数 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 训练所有模型
        print("=" * 50)
        
        print("Training Windowed ESN...")
        WindowedESN_model, WindowedESN_avg_loss, WindowedESN_total_loss = train_windowed_esn()
        
        print("Training Continuous ESN...")
        ContinuousESN_model, ContinuousESN_avg_loss, ContinuousESN_total_loss = train_continuous_esn()
        
        # 绘制损失对比
        # 评估所有模型
        seq_length_for_eval = 25
        
        print("Evaluating Windowed ESN...")
        t_pred_w_esn, y_true_w_esn, y_pred_w_esn = evaluate_windowed_esn(
            model=WindowedESN_model, seq_len=seq_length_for_eval, device="cpu", total_points=500)
        
        print("Evaluating Continuous ESN...")
        t_pred_s_esn, y_true_s_esn, y_pred_s_esn = evaluate_continuous_esn(
            model=ContinuousESN_model, device="cpu", total_points=500)
        
        # 绘制预测结果
        print("Plotting predictions...")
        plot_predictions(t_pred_w_esn, y_true_w_esn, y_pred_w_esn, "Windowed ESN Prediction vs True Sine")
        plot_predictions(t_pred_s_esn, y_true_s_esn, y_pred_s_esn, "Continuous ESN Prediction vs True Sine")
        
        print("All experiments completed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()