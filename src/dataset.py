import numpy as np
import torch
from torch.utils.data import Dataset

class Seq2SeqSlidingDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0]
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        return x, y

def build_sliding_windows(
    data_scaled_values: np.ndarray,
    extra_feats: np.ndarray,
    seq_len: int,
    pred_len: int
):
    C = data_scaled_values.shape[1]
    T = data_scaled_values.shape[0]
    F_extra = 0 if extra_feats is None else extra_feats.shape[1]

    X_list, Y_list = [], []
    for i in range(T - seq_len - pred_len + 1):
        past_y = data_scaled_values[i:i+seq_len, :]
        fut_y  = data_scaled_values[i+seq_len:i+seq_len+pred_len, :]

        if F_extra > 0:
            past_e = extra_feats[i:i+seq_len, :]
            x = np.concatenate([past_y, past_e], axis=1)
        else:
            x = past_y

        X_list.append(x)
        Y_list.append(fut_y.reshape(-1))

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y
