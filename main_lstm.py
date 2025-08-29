# -*- coding: utf-8 -*-
"""
多目标（6区县）宽表流水线：
python main_lstm.py --config config/config.yaml --run_name best_retrain1 \
  --use_fill_calendar --use_scaler \
  --do_train --do_eval --do_predict --do_plot
"""
from __future__ import annotations
import argparse
import yaml
from pathlib import Path
from typing import Union, List
import pandas as pd
import torch

from tools.seed import set_seed
from tools.io import ensure_dir
from tools.plot_combined import plot_per_series
from tools.evaluator import evaluate_test
from tools.predictor import forecast_future

from preprocess.preprocessing import (
    add_time_features, fill_missing_by_calendar, MinMaxScalerDict
)
from preprocess.slicing import create_sliding_windows_multi
from dataset.timeseries_dataset import TimeSeriesDataset
from models.lstm import LSTMForecaster
from training.trainer import train_loop

def load_cfg(p: Union[str, Path]) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _derive_extra_cols_from_df(df: pd.DataFrame) -> List[str]:
    cols = []
    if "is_holiday" in df.columns: cols.append("is_holiday")
    if "is_month_end" in df.columns: cols.append("is_month_end")
    if "weekday_sin" in df.columns and "weekday_cos" in df.columns:
        cols += ["weekday_sin", "weekday_cos"]
    elif "weekday" in df.columns:
        cols += ["weekday"]
    if "month_sin" in df.columns and "month_cos" in df.columns:
        cols += ["month_sin", "month_cos"]
    elif "month" in df.columns:
        cols += ["month"]
    return cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_plot", action="store_true")
    parser.add_argument("--use_fill_calendar", action="store_true")
    parser.add_argument("--use_scaler", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.run_name: cfg["run_name"] = args.run_name
    set_seed(cfg["train"].get("seed", 42))

    out_dir = Path(cfg["paths"]["outputs"]) / "runs" / cfg["run_name"]
    ensure_dir(out_dir)

    # 读取宽表（date + series_cols）
    df = pd.read_csv(cfg["data"]["csv_path"])

    # （可选）缺失值修复：宽表模式 group_col=None / target_col=None
    if args.use_fill_calendar:
        df = fill_missing_by_calendar(
            df,
            date_col=cfg["data"]["date_col"],
            group_col=None,
            target_col=None,
            freq=cfg["data"].get("freq", "D"),
        )

    # 时间特征
    df = add_time_features(
        df,
        date_col=cfg["data"]["date_col"],
        add_time_features=cfg["data"].get("add_time_features", True),
        weekday_cyclical=cfg["data"].get("weekday_cyclical", False),
        month_cyclical=cfg["data"].get("month_cyclical", True),
    )

    series_cols = list(cfg["data"]["series_cols"])
    extra_cols = _derive_extra_cols_from_df(df)

    # （可选）缩放（series+extra 一起）
    scaler = None
    feat_total = series_cols + extra_cols
    if args.use_scaler:
        scaler = MinMaxScalerDict(feat_total).fit(df)
        scaler.save(Path(out_dir) / "logs" / "scaler.json")
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df

    # 滑窗（多目标）
    X, Y, pred_idx = create_sliding_windows_multi(
        df=df_scaled,
        date_col=cfg["data"]["date_col"],
        series_cols=series_cols,
        extra_feature_cols=extra_cols,
        seq_len=cfg["window"]["seq_len"],
        pred_len=cfg["window"]["pred_len"],
        stride=cfg["window"]["stride"],
        return_index=True,
    )

    # 切分
    N = len(X)
    tr = int(N * cfg["split"]["train_ratio"])
    va = int(N * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    train_ds = TimeSeriesDataset(X[:tr], Y[:tr])
    val_ds   = TimeSeriesDataset(X[tr:va], Y[tr:va])
    test_ds  = TimeSeriesDataset(X[va:],  Y[va:])
    test_pred_dates = pred_idx[va:]

    # 模型（输入维自动推断；输出=pred_len*D）
    input_size = X.shape[2]
    D = len(series_cols)
    model = LSTMForecaster(
        input_size=input_size,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        pred_len=cfg["window"]["pred_len"],
        n_targets=D,
    )

    # 训练 / 加载
    if args.do_train:
        _ = train_loop(model, train_ds, val_ds, cfg, out_dir, test_size=len(test_ds))
    else:
        ckpt_path = Path(out_dir) / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到已训练权重：{ckpt_path}，请先 --do_train")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    # 评估（输出每日×多列 CSV）
    if args.do_eval:
        metrics, _df = evaluate_test(model, test_ds, test_pred_dates, cfg, out_dir, scaler=scaler if args.use_scaler else None)

    # 未来多目标预测（输出 y_pred_* / 可选 y_pred_inv_*）
    if args.do_predict:
        _future = forecast_future(
            model,
            df_scaled.copy(),
            cfg,
            out_dir,
            scaler=scaler if args.use_scaler else None,
        )
        print("Future forecast saved at:", Path(out_dir) / "predictions" / "future_forecast.csv")

    # 单区县可视化
    if args.do_plot:
        test_csv   = Path(out_dir) / "predictions" / "test_predictions.csv"
        future_csv = Path(out_dir) / "predictions" / "future_forecast.csv"
        fig_dir    = Path(cfg["paths"]["figures"]) / cfg['run_name']
        plot_per_series(test_csv, future_csv, fig_dir)

if __name__ == "__main__":
    main()
