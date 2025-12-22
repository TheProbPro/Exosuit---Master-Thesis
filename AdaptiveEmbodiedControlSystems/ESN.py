import torch
import torch.nn as nn
import numpy as np

class ESN(nn.Module):
    """
    Echo State Network for time series prediction
    """
    def __init__(self, input_size, reservoir_size, output_size, 
                 spectral_radius=0.9, leaking_rate=0.7, connectivity=0.1):
        super(ESN, self).__init__()
        
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate
        
        # Initialize input weights as nn.Parameter for proper device handling
        self.W_in = nn.Parameter(torch.randn(reservoir_size, input_size) * 1.0, requires_grad=False)
        
        # Initialize reservoir weights (sparse) as nn.Parameter
        W_res_init = self._initialize_reservoir(reservoir_size, connectivity, spectral_radius)
        self.W_res = nn.Parameter(W_res_init, requires_grad=False)
        
        # Output weights (will be trained)
        self.W_out = nn.Linear(reservoir_size, output_size, bias=True)
        
    def _initialize_reservoir(self, size, connectivity, spectral_radius):
        # Create sparse reservoir matrix
        W = torch.randn(size, size)
        # Apply sparsity
        mask = torch.rand(size, size) > connectivity
        W[mask] = 0
        
        # Normalize to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        current_spectral_radius = torch.max(torch.abs(eigenvalues))
        W = W * (spectral_radius / current_spectral_radius)
        
        return W
    
    def forward(self, x, state=None):
        batch_size, seq_len, input_dim = x.shape
        
        if state is None:
            state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        
        reservoir_states = []
        current_state = state
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Update reservoir state
            input_term = torch.mm(x_t, self.W_in.T)
            reservoir_term = torch.mm(current_state, self.W_res.T)
            
            new_state = (1 - self.leaking_rate) * current_state + \
                       self.leaking_rate * torch.tanh(input_term + reservoir_term)
            
            reservoir_states.append(new_state.unsqueeze(1))
            current_state = new_state
        
        # Stack all reservoir states
        reservoir_states = torch.cat(reservoir_states, dim=1)
        
        # Output layer
        outputs = self.W_out(reservoir_states)
        
        return outputs, current_state

class WindowedESN(nn.Module):
    """
    ESN for window-based prediction (similar to windowed LSTM)
    Only outputs prediction for the last time step
    """
    def __init__(self, input_size, reservoir_size, output_size, 
                 spectral_radius=0.9, leaking_rate=0.7, connectivity=0.1):
        super(WindowedESN, self).__init__()
        
        self.esn = ESN(input_size, reservoir_size, output_size, 
                      spectral_radius, leaking_rate, connectivity)
        
    def forward(self, x, state=None):
        # x: (batch, seq_len, input_size)
        outputs, final_state = self.esn(x, state)
        
        # Only use the last time step for prediction (like windowed LSTM)
        last_output = outputs[:, -1, :]
        
        return last_output