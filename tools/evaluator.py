# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tools.metrics import mae, mse, rmse, mape_safe
from tools.io import ensure_dir, save_csv, save_json
from typing import Optional
import logging

def _derive_extra_cols(cfg_data: dict) -> List[str]:
    if not cfg_data.get("add_time_features", True):
        return []
    cols = ["is_holiday", "is_month_end"]
    if cfg_data.get("weekday_cyclical", False):
        cols += ["weekday_sin", "weekday_cos"]
    else:
        cols += ["weekday"]
    if cfg_data.get("month_cyclical", True):
        cols += ["month_sin", "month_cos"]
    else:
        cols += ["month"]
    return cols

def evaluate_test(model, test_ds, pred_dates, cfg: dict, out_dir, scaler: Optional[object] = None, logger: Optional[logging.Logger] = None):
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    D = len(cfg["data"]["series_cols"])
    P = cfg["window"]["pred_len"]
    extra_cols = _derive_extra_cols(cfg["data"])

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for x, y_flat in loader:
            x = x.to(device)                              # [B, L, F]
            pred_flat = model(x).cpu().numpy()            # [B, P*D]
            y_flat = y_flat.cpu().numpy()                 # [B, P*D]
            y_true_list.append(y_flat)
            y_pred_list.append(pred_flat)

    y_true_flat = np.concatenate(y_true_list, axis=0)     # [N, P*D]
    y_pred_flat = np.concatenate(y_pred_list, axis=0)     # [N, P*D]

    # 还原形状便于落盘
    y_true = y_true_flat.reshape(-1, P, D)   # [N, P, D]
    y_pred = y_pred_flat.reshape(-1, P, D)

    # ---------- 反归一化（若提供 scaler） ----------
    if scaler is not None:
        n_rows = y_true.shape[0] * y_true.shape[1]
        series_cols = list(cfg["data"]["series_cols"])
        place_cols = series_cols + extra_cols

        true_df = pd.DataFrame(0.0, index=range(n_rows), columns=place_cols)
        for j, name in enumerate(series_cols):
            true_df[name] = y_true.reshape(-1, D)[:, j]
        true_inv = scaler.inverse_transform(true_df)
        y_true = true_inv[series_cols].values.reshape(-1, P, D)

        pred_df = pd.DataFrame(0.0, index=range(n_rows), columns=place_cols)
        for j, name in enumerate(series_cols):
            pred_df[name] = y_pred.reshape(-1, D)[:, j]
        pred_df[series_cols] = pred_df[series_cols].clip(0.0, 1.0)
        pred_inv = scaler.inverse_transform(pred_df)
        y_pred = pred_inv[series_cols].clip(lower=0.0).values.reshape(-1, P, D)

    metrics = {
        "mae":  mae(y_true.ravel(), y_pred.ravel()),
        "mse":  mse(y_true.ravel(), y_pred.ravel()),
        "rmse": rmse(y_true.ravel(), y_pred.ravel()),
        "mape_safe": mape_safe(y_true.ravel(), y_pred.ravel()),
    }

    # ---------- 落盘（仅真实尺度） ----------
    rows = []
    scols = cfg["data"]["series_cols"]
    for i in range(y_true.shape[0]):
        for t in range(P):
            row = {"index": i, "step": t + 1, "date": pd.to_datetime(pred_dates[i, t]).date()}
            for j, name in enumerate(scols):
                row[f"y_true_{name}"] = y_true[i, t, j]
                row[f"y_pred_{name}"] = y_pred[i, t, j]
            rows.append(row)
    df_pred = pd.DataFrame(rows)

    out_dir = Path(out_dir); ensure_dir(out_dir / "predictions")
    save_csv(out_dir / "predictions" / "test_predictions.csv", df_pred)
    ensure_dir(out_dir / "logs")
    save_json(metrics, out_dir / "logs" / "eval_metrics.json")
    if logger is not None:
        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(f"Saved test predictions to: {out_dir / 'predictions' / 'test_predictions.csv'}")
    else:
        print("Evaluation metrics:", metrics)
    return metrics, df_pred
