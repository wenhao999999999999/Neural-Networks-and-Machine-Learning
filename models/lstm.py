# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    输入:  [B, L, F]
    输出:  [B, pred_len * D]  (D = n_targets)
    """
    def __init__(self, input_size, hidden_dim, num_layers, dropout, pred_len, n_targets):
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.lstm = nn.LSTM(
            input_size, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, pred_len * n_targets)

    def forward(self, x):
        out, _ = self.lstm(x)    # [B, L, H]
        h = out[:, -1, :]        # [B, H]
        y = self.fc(h)           # [B, P*D]
        return y
