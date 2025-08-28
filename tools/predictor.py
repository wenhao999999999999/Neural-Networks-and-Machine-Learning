# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import torch
from tools.io import ensure_dir, save_csv

# 无库时，周末当休息日
try:
    import chinese_calendar as cncal
except Exception:
    cncal = None

def _is_holiday_or_rest(day: pd.Timestamp) -> int:
    if cncal is not None:
        try:
            return int(not cncal.is_workday(day))
        except Exception:
            pass
    return int(day.weekday() >= 5)

def _compute_extra_row(next_date: pd.Timestamp, use_weekday_cyc: bool, use_month_cyc: bool):
    row = {}
    row["is_holiday"] = _is_holiday_or_rest(next_date)
    row["is_month_end"] = int(pd.Timestamp(next_date).is_month_end)
    wd = int(pd.Timestamp(next_date).weekday())
    if use_weekday_cyc:
        row["weekday_sin"] = np.sin(2 * np.pi * wd / 7.0)
        row["weekday_cos"] = np.cos(2 * np.pi * wd / 7.0)
    else:
        row["weekday"] = wd
    m = int(pd.Timestamp(next_date).month)
    if use_month_cyc:
        row["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12.0)
        row["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12.0)
    else:
        row["month"] = m
    return row

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

def forecast_future(
    model,
    last_context_df: pd.DataFrame,  # 已加时间特征（并可能缩放）的宽表
    cfg: dict,
    out_dir: Union[str, Path],
    scaler: Optional[object] = None,
):
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    model = model.to(device); model.eval()

    series_cols = list(cfg["data"]["series_cols"])
    extra_cols = _derive_extra_cols(cfg["data"])
    seq_len = cfg["window"]["seq_len"]
    horizon = cfg["predict"]["horizon"]
    date_col = cfg["data"]["date_col"]

    feat_cols_total = series_cols + extra_cols
    df_sorted = last_context_df.sort_values(date_col)
    cur_x = df_sorted[feat_cols_total].values[-seq_len:, :].astype(np.float32)

    base_date = pd.to_datetime(df_sorted[date_col]).iloc[-1]
    D = len(series_cols)

    dates, preds_mat = [], []
    for step in range(horizon):
        xx = torch.tensor(cur_x[None, :, :], dtype=torch.float32).to(device)  # [1, L, F]
        with torch.no_grad():
            y_flat = model(xx).cpu().numpy()[0]  # [P*D]
        y_next = y_flat[:D]                      # 一步递归
        preds_mat.append(y_next.copy())

        next_date = base_date + pd.Timedelta(days=step + 1)
        dates.append(next_date.date())
        extra_next = _compute_extra_row(
            next_date,
            use_weekday_cyc=cfg["data"].get("weekday_cyclical", False),
            use_month_cyc=cfg["data"].get("month_cyclical", True),
        )
        extra_vec = [extra_next[c] for c in extra_cols]
        next_row = np.concatenate([y_next.astype(np.float32), np.array(extra_vec, dtype=np.float32)], axis=0)
        cur_x = np.concatenate([cur_x[1:, :], next_row[None, :]], axis=0)

    preds_mat = np.asarray(preds_mat, dtype=np.float32)  # [H, D]
    out = {"date": dates}
    for j, name in enumerate(series_cols):
        out[f"y_pred_{name}"] = preds_mat[:, j]
    df_future = pd.DataFrame(out)


    # 反归一化修复：提供 index & 列集合，并先在归一化空间 clip
    if scaler is not None:
        n = len(df_future)
        place_cols = series_cols + extra_cols
        place = pd.DataFrame(0.0, index=range(n), columns=place_cols)
        for name in series_cols:
            place[name] = df_future[f"y_pred_{name}"].values
        place[series_cols] = place[series_cols].clip(0.0, 1.0)
        place_inv = scaler.inverse_transform(place)
        for name in series_cols:
            df_future[f"y_pred_inv_{name}"] = place_inv[name].clip(lower=0.0).values

