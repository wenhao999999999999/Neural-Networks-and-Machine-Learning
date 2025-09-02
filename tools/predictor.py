# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import torch
from tools.io import ensure_dir, save_csv
from typing import Optional
import logging

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
    logger: Optional[logging.Logger] = None,
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
        # 注意：如果训练时做了缩放，则这里递归拼接到输入序列中的“时间特征”也必须按相同的缩放处理，
        # 否则模型在推理时会接收到未缩放的特征，导致数值发散、预测不合理。
        if scaler is not None and hasattr(scaler, "stats"):
            scaled_extras: List[float] = []
            for c in extra_cols:
                v = float(extra_next[c])
                try:
                    lo = float(scaler.stats[c]["min"])
                    hi = float(scaler.stats[c]["max"])
                except Exception:
                    lo, hi = 0.0, 1.0
                denom = (hi - lo) if (hi - lo) != 0 else 1.0
                vv = (v - lo) / denom
                # 按训练时一致进行裁剪
                if getattr(scaler, "clip", True):
                    vv = max(0.0, min(1.0, vv))
                scaled_extras.append(vv)
            extra_vec = np.array(scaled_extras, dtype=np.float32)
        else:
            extra_vec = np.array([extra_next[c] for c in extra_cols], dtype=np.float32)

        # y_next 已经是“缩放空间”的一阶预测（因为模型输入是缩放后的特征）；
        # 这里拼接到下一时刻上下文仍需保持缩放空间。
        next_row = np.concatenate([y_next.astype(np.float32), extra_vec], axis=0)
        cur_x = np.concatenate([cur_x[1:, :], next_row[None, :]], axis=0)

    preds_mat = np.asarray(preds_mat, dtype=np.float32)  # [H, D]

    if scaler is not None:
        n = preds_mat.shape[0]
        place_cols = series_cols + extra_cols
        place = pd.DataFrame(0.0, index=range(n), columns=place_cols)
        for j, name in enumerate(series_cols):
            place[name] = preds_mat[:, j]
        place[series_cols] = place[series_cols].clip(0.0, 1.0)
        place_inv = scaler.inverse_transform(place)
        preds_mat = place_inv[series_cols].clip(lower=0.0).values

    out = {"date": dates}
    for j, name in enumerate(series_cols):
        out[f"y_pred_{name}"] = preds_mat[:, j]
    df_future = pd.DataFrame(out)

    out_dir = Path(out_dir)
    ensure_dir(out_dir / "predictions")
    save_csv(out_dir / "predictions" / "future_forecast.csv", df_future)
    if logger is not None:
        logger.info(f"Saved future forecast to: {out_dir / 'predictions' / 'future_forecast.csv'}")
    return df_future
