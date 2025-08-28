# -*- coding: utf-8 -*-
"""
示例：
python tuning/tune_lstm.py --config config/config.yaml --n_trials 30 --run_name tune_20250828
"""
import argparse, yaml, optuna
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from preprocess.preprocessing import add_time_features, MinMaxScalerDict, fill_missing_by_calendar
from preprocess.slicing import create_sliding_windows
from dataset.timeseries_dataset import TimeSeriesDataset
from models.lstm import LSTMForecaster
from training.trainer import EarlyStopper
from torch.utils.data import DataLoader

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def objective(trial, cfg):
    # 超参空间
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_len    = trial.suggest_categorical("seq_len", [7, 15, 21, 30])
    pred_len   = cfg["window"]["pred_len"]  # 固定评估步长

    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    # 数据
    df = pd.read_csv(cfg["data"]["csv_path"])
    df = fill_missing_by_calendar(df, cfg["data"]["date_col"], cfg["data"]["group_col"], cfg["data"]["target_col"], cfg["data"].get("freq","D"))
    df = add_time_features(df, cfg["data"]["date_col"])

    # 可选：MinMax 缩放（包含目标列）
    feats_all = cfg["data"]["feature_cols"] + [cfg["data"]["target_col"]]
    scaler = MinMaxScalerDict(feats_all).fit(df)
    df_scaled = scaler.transform(df)

    # 滑窗
    X, Y, _, _ = create_sliding_windows(
        df=df_scaled,
        group_col=cfg["data"]["group_col"],
        date_col=cfg["data"]["date_col"],
        feature_cols=cfg["data"]["feature_cols"] + [cfg["data"]["target_col"]],
        target_col=cfg["data"]["target_col"],
        seq_len=seq_len,
        pred_len=pred_len,
        stride=cfg["window"]["stride"],
        return_index=False,
    )
    N = len(X)
    tr = int(N * cfg["split"]["train_ratio"])
    va = int(N * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    train_ds = TimeSeriesDataset(X[:tr], Y[:tr])
    val_ds   = TimeSeriesDataset(X[tr:va], Y[tr:va])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 模型
    model = LSTMForecaster(
        input_size=cfg["model"]["input_size"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    early = EarlyStopper(patience=cfg["train"]["patience"], min_delta=float(cfg["train"].get("min_delta", 0.0)))
    best_val = np.inf

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train(); tl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tl += loss.item() * len(x)
        tl /= len(train_loader.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                vl += loss_fn(pred, y).item() * len(x)
        vl /= len(val_loader.dataset)

        if vl < best_val:
            best_val = vl
        early.step(vl)
        if early.stop:
            break

    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="tuning")
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, cfg), n_trials=args.n_trials)
    print("Best trial:", study.best_trial.number, "| val_mse:", study.best_value)
    print("Best params:", study.best_params)

    # 保存结果
    out = Path(cfg["paths"]["outputs"]) / "runs" / args.run_name / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_params.json", "w", encoding="utf-8") as f:
        import json; json.dump(study.best_params, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
