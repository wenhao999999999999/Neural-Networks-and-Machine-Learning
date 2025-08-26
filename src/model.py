import torch
import torch.nn as nn

class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int, dropout: float,
                 pred_len: int, num_series: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * num_series),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.head(h)
        return y
