import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error, r2_score
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

def generate_test_sine(total_points=1000, noise_std=0.0, device="cpu"):
    """
    Create a clean (or slightly noisy) sine wave to test the trained model.
    Returns:
        signal: (T, 1) tensor
        t_axis: (T,)  tensor of time indices (float)
    """
    t_axis = torch.linspace(0, 20*np.pi, steps=total_points)
    clean = torch.sin(t_axis)
    if noise_std > 0:
        clean = clean + noise_std * torch.randn_like(clean)

    signal = clean.unsqueeze(-1).to(device)  # (T,1)
    return signal, t_axis

class SineDataset(Dataset):
    '''A dataset of sine wave sequences for training the LSTM.'''
    def __init__(self, total_points=5000, seq_len=50):
        super().__init__()
        self.seq_len = seq_len

        # make a long sine wave with some frequency + maybe noise
        t = torch.linspace(0, 20*np.pi, steps=total_points)  # 10-ish periods
        full_signal = torch.sin(t)  # shape (total_points,)

        # build samples
        X_list = []
        y_list = []
        for i in range(total_points - seq_len - 1):
            window = full_signal[i : i+seq_len]         # length seq_len
            target = full_signal[i+seq_len]             # the "next" point
            X_list.append(window.unsqueeze(-1))         # (seq_len, 1)
            y_list.append(target.unsqueeze(-1))         # (1,)

        self.X = torch.stack(X_list, dim=0)  # (num_samples, seq_len, 1)
        self.y = torch.stack(y_list, dim=0)  # (num_samples, 1)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ContinuousSineDataset(Dataset):
    """
    A continuous sine-like signal that we will walk through sequentially.
    For streaming training, we won't index randomly. We'll just grab [0..T-1].
    """
    def __init__(self, total_points=5000):
        super().__init__()

        t = torch.linspace(0, 20*np.pi, steps=total_points)
        signal = torch.sin(t)

        # optional realism: add a little noise so it's not trivial
        noise = 0.05 * torch.randn_like(signal)
        signal = signal + noise  # shape (T,)

        # store as (T, 1) so it's [time, feature_dim]
        self.signal = signal.unsqueeze(-1)

    def __len__(self):
        return self.signal.size(0)

    def __getitem__(self, idx):
        # we won't actually rely on random indexing in the trainer,
        # but we'll keep this here to satisfy Dataset API expectations.
        return self.signal[idx]


def train_moving_window_lstm():
    # hyperparameters
    seq_length = 25
    hiddenstate = 64
    num_layers = 1
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_loss_array = np.array([])
    total_loss_array = np.array([])

    # data
    dataset = SineDataset(total_points=5000, seq_len=seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = LSTM.LSTMModel(input_size=1, hidden_size=hiddenstate, output_size=1, num_layers=num_layers, batch_first=True).to(device)

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



def train_continuous_lstm():
    # Hyperparameters
    hidden_state = 64
    num_layers = 1
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 20
    tbptt_steps = 50      # how many single-sample steps before backprop
    total_points = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_loss_array = np.array([])
    total_loss_array = np.array([])


    # Data
    dataset = ContinuousSineDataset(total_points=5000)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # take the whole thing as one long tensor (T, 1)
    full_signal = dataset.signal.to(device)
    T = full_signal.shape[0]

    # Model
    model = LSTM.StreamingLSTM(input_size=1, hidden_size=hidden_state, output_size=1, num_layers=num_layers, batch_first=True).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        h = None
        c = None

        running_loss = 0.0

        # for truncated BPTT bookkeeping
        step_in_chunk = 0
        chunk_losses = []

        # We'll iterate from t = 0 to T-2, because we predict sample t+1
        for t in range(T - 1):
            # current input sample x_t -> (1,1,1)
            x_t = full_signal[t].view(1, 1, 1)        # batch=1, seq_len=1, feat=1
            y_target = full_signal[t+1].view(1, 1, 1) # predict next sample

            # forward pass on ONE timestep, carrying h,c
            preds, (h, c) = model(x_t, (h, c))
            # preds: (1,1,1)

            loss_t = criterion(preds, y_target)
            chunk_losses.append(loss_t)
            step_in_chunk += 1

            # if we've collected tbptt_steps steps, backprop through that chunk
            if step_in_chunk == tbptt_steps:
                loss_chunk = torch.stack(chunk_losses).mean()

                optimizer.zero_grad()
                loss_chunk.backward()
                optimizer.step()

                # VERY IMPORTANT:
                # cut the graph here but keep the numeric hidden state
                h = h.detach()
                c = c.detach()

                running_loss += loss_chunk.item()
                # reset chunk accumulator
                chunk_losses = []
                step_in_chunk = 0

        # handle leftover chunk at the end of the sequence if it didn't
        # line up perfectly with tbptt_steps
        if step_in_chunk > 0:
            loss_chunk = torch.stack(chunk_losses).mean()
            optimizer.zero_grad()
            loss_chunk.backward()
            optimizer.step()

            h = h.detach()
            c = c.detach()

            running_loss += loss_chunk.item()

        # bookkeeping like your script
        total_loss_array = np.append(total_loss_array, running_loss)

        # avg loss per "chunk" this epoch (T // tbptt_steps chunks approx)
        denom = max(1, (T // tbptt_steps))
        avg_loss = running_loss / denom
        average_loss_array = np.append(average_loss_array, avg_loss)

        print(f"epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.6f}")

    return model, average_loss_array, total_loss_array

@torch.no_grad()
def evaluate_windowed_lstm(model, seq_len, device="cpu", total_points=1000):
    """
    Runs the trained windowed LSTM model on a fresh sine wave and returns:
    - t_pred: time indices that correspond to the predicted points
    - y_true: ground truth sine values at those indices
    - y_pred: model's predicted values
    """

    model = model.to(device)
    model.eval()

    # 1) make test sine wave
    signal, t_axis = generate_test_sine(total_points=total_points, noise_std=0.0, device=device)
    # signal shape: (T,1)

    T = signal.shape[0]

    # we'll build all windows in one big batch for fast eval
    X_list = []
    for i in range(T - seq_len - 1):
        window = signal[i : i+seq_len]            # (seq_len,1)
        X_list.append(window.unsqueeze(0))        # (1,seq_len,1)

    if len(X_list) == 0:
        raise ValueError("total_points too small vs seq_len during evaluation")

    X_batch = torch.cat(X_list, dim=0)            # (N, seq_len, 1)

    # 2) run model in batch
    preds = model(X_batch)                        # (N,1) because your model outputs only last step
    # reshape to (N,)
    y_pred = preds.squeeze(-1).squeeze(-1).cpu().numpy()

    # 3) build ground truth targets aligned with predictions
    # each window i predicts point at index (i+seq_len)
    target_indices = torch.arange(seq_len, seq_len + len(y_pred))
    y_true = signal[target_indices, 0].cpu().numpy()
    t_pred = t_axis[target_indices].cpu().numpy()

    return t_pred, y_true, y_pred

@torch.no_grad()
def evaluate_streaming_lstm(model, device="cpu", total_points=1000):
    """
    Runs the trained StreamingLSTM model sample-by-sample on a fresh sine wave.
    Returns:
    - t_pred: time indices for predicted points (start at 1 because we predict t+1)
    - y_true: ground truth sine at those indices
    - y_pred: model prediction for those indices
    """

    model = model.to(device)
    model.eval()

    # 1) test sine wave
    signal, t_axis = generate_test_sine(total_points=total_points, noise_std=0.0, device=device)
    # signal: (T,1)
    T = signal.shape[0]

    # 2) init hidden state for batch=1
    h = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
    c = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)

    preds_list = []
    t_idx_list = []

    # iterate through sequence
    # we go up to T-2 because we predict t+1 and need it to exist
    for t in range(T - 1):
        x_t = signal[t].view(1, 1, -1)  # shape (1,1,input_size)
        pred, (h, c) = model(x_t, (h, c))  # pred: (1,1,output_size)

        # save the predicted value for index t+1
        preds_list.append(pred.item())          # scalar if output_size==1
        t_idx_list.append(t + 1)

        # IMPORTANT in eval: detach hidden to prevent graph buildup
        h = h.detach()
        c = c.detach()

    # build arrays
    t_idx_tensor = torch.tensor(t_idx_list, device=device)
    t_pred = t_axis[t_idx_tensor].cpu().numpy()          # time for t+1
    y_true = signal[t_idx_tensor, 0].cpu().numpy()       # ground truth at t+1
    y_pred = np.array(preds_list)

    return t_pred, y_true, y_pred

def plot_predictions(t_pred, y_true, y_pred, title):
    plt.figure(figsize=(10,4))
    plt.plot(t_pred, y_true, label="Ground truth", linewidth=2)
    plt.plot(t_pred, y_pred, label="LSTM prediction", linestyle='--')
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.title(title, fontsize=12)
    plt.xlim([t_pred[0], t_pred[-1]])
    plt.legend(fontsize=10, loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    print("Training Windowed LSTM...")
    WindowedLSTM, Windowed_avg_loss, Windowed_total_loss = train_moving_window_lstm()
    # Plotting the avg loss and the windowed total loss arrays in two subplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Windowed_avg_loss, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss over Epochs (Windowed LSTM)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Windowed_total_loss, label='Total Loss per Epoch', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss over Epochs (Windowed LSTM)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Training Continuous LSTM...")
    ContinuousLSTM, Continuous_avg_loss, Continuous_total_loss = train_continuous_lstm()
    # Plotting the avg loss and the windowed total loss arrays in two subplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Continuous_avg_loss, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss over Epochs (Continuous LSTM)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Continuous_total_loss, label='Total Loss per Epoch', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss over Epochs (Continuous LSTM)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Evaluate and plot predictions for both models ----

    # 1. Windowed model evaluation
    seq_length_for_eval = 25  # must match seq_length used to train your windowed model
    t_pred_w, y_true_w, y_pred_w = evaluate_windowed_lstm(
        model=WindowedLSTM,
        seq_len=seq_length_for_eval,
        device="cpu",
        total_points=1000
    )
    plot_predictions(
        t_pred_w,
        y_true_w,
        y_pred_w,
        title="Windowed LSTM Prediction vs True Sine"
    )

    # 2. Streaming model evaluation
    t_pred_s, y_true_s, y_pred_s = evaluate_streaming_lstm(
        model=ContinuousLSTM,
        device="cpu",
        total_points=1000
    )
    plot_predictions(
        t_pred_s,
        y_true_s,
        y_pred_s,
        title="Streaming LSTM Prediction vs True Sine"
    )