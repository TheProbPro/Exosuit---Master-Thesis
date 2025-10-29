import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class StreamingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        # hidden (optional): tuple(h, c) where
        #   h: (num_layers, batch, hidden_size)
        #   c: (num_layers, batch, hidden_size)

        if hidden is None or hidden[0] is None or hidden[1] is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h0, c0 = hidden

        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out: (batch, seq_len, hidden_size)

        preds = self.fc(out)
        # preds: (batch, seq_len, output_size)
        # Now you have one prediction per timestep.

        return preds, (hn, cn)


# Implementation without nn.LSTM
# class LSTMCell(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(LSTMCell, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
        
#         # Input gate components
#         self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
#         # Forget gate components
#         self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
#         # Cell gate components
#         self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
#         # Output gate components
#         self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.init_weights()

#     def init_weights(self):
#         for param in self.parameters():
#             nn.init.uniform_(param, -0.1, 0.1)
        
#     def forward(self, x, hidden):
#         h_prev, c_prev = hidden

#         i_t = torch.sigmoid(x @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
#         f_t = torch.sigmoid(x @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
#         g_t = torch.tanh(x @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
#         o_t = torch.sigmoid(x @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)
        
#         c_t = f_t * c_prev + i_t * g_t
#         h_t = o_t * torch.tanh(c_t)
        
#         return h_t, (h_t, c_t)


# class LSTM(nn.Module):
#     def __init__(self, input_window_size, hidden_size, num_layers):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.cells = nn.ModuleList([LSTMCell(input_window_size, hidden_size) if i == 0 
#                                     else LSTMCell(hidden_size, hidden_size) 
#                                     for i in range(num_layers)])
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
#         c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
#         for t in range(seq_len):
#             x_t = x[:, t, :]
#             for i, cell in enumerate(self.cells):
#                 h[i], (h[i], c[i]) = cell(x_t, (h[i], c[i]))
#                 x_t = h[i]
        
#         out = self.fc(h[-1])
#         return out