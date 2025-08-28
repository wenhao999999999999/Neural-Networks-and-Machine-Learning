# -*- coding: utf-8 -*-
from typing import List, Optional
import numpy as np
import pandas as pd

def create_sliding_windows_multi(
    df: pd.DataFrame,
    date_col: str,
    series_cols: List[str],
    extra_feature_cols: Optional[List[str]],
    seq_len: int,
    pred_len: int,
    stride: int = 1,
    return_index: bool = True,
):
    """
    多目标（D=len(series_cols)）宽表滑窗：
    X: [N, seq_len, F]，F = D + len(extra_feature_cols)
    Y: [N, pred_len*D]  （训练时展平）
    pred_index: [N, pred_len] 的日期
    """
    series_cols = list(series_cols)
    extra_feature_cols = list(extra_feature_cols or [])
    dff = df.sort_values(date_col).reset_index(drop=True)

    X_all = dff[series_cols + extra_feature_cols].values.astype(np.float32)
    Y_all = dff[series_cols].values.astype(np.float32)  # 仅系列列监督
    dates = pd.to_datetime(dff[date_col]).values

    T = len(dff); D = len(series_cols)
    X_list, Y_list, pred_idx = [], [], []
    for s in range(0, T - seq_len - pred_len + 1, stride):
        x = X_all[s: s + seq_len, :]                       # [L, F]
        y = Y_all[s + seq_len: s + seq_len + pred_len, :]  # [P, D]
        X_list.append(x)
        Y_list.append(y.reshape(-1))
        if return_index:
            pred_idx.append(dates[s + seq_len: s + seq_len + pred_len])

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    pred_index = np.asarray(pred_idx, dtype="datetime64[ns]") if return_index else None
    return X, Y, pred_index
