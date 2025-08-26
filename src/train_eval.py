import os
import argparse
import time
import yaml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import set_seed, ensure_dir, save_json, pick_device
from metrics import per_series_metrics
from dataset import Seq2SeqSlidingDataset, build_sliding_windows
from model import LSTMSeq2Seq
from data_prep import load_and_prepare, fill_missing_by_rule, build_feature_table, MinMaxScalerPerColumn

def choose_feat_cols(df_part: pd.DataFrame):
    """
    只选择用于训练的时间特征列（顺序固定）：
    is_holiday, weekday_sin/cos 或 weekday, month_sin/cos 或 month, is_month_end
    """
    cols = []
    if 'is_holiday' in df_part.columns: cols += ['is_holiday']
    if 'weekday_sin' in df_part.columns and 'weekday_cos' in df_part.columns:
        cols += ['weekday_sin','weekday_cos']
    elif 'weekday' in df_part.columns:
        cols += ['weekday']
    if 'month_sin' in df_part.columns and 'month_cos' in df_part.columns:
        cols += ['month_sin','month_cos']
    elif 'month' in df_part.columns:
        cols += ['month']
    if 'is_month_end' in df_part.columns: cols += ['is_month_end']
    return cols

def make_loaders(df_feat: pd.DataFrame, y_cols, cfg):
    seq_len = cfg['seq_len']; pred_len = cfg['pred_len']
    val_len = cfg['val_len'];  test_len = cfg['test_len']

    assert val_len >= pred_len, (
        f"val_len({val_len}) 必须 >= pred_len({pred_len})，否则验证集无样本。"
    )

    # 划分：|-------- train --------|--- val ---|--- test ---|
    T = len(df_feat)
    train_end = T - val_len - test_len
    val_end   = T - test_len

    df_train = df_feat.iloc[:train_end]
    start_val_ctx = max(0, train_end - seq_len)
    df_val_ctx = df_feat.iloc[start_val_ctx:val_end]
    df_test  = df_feat.iloc[val_end:]

    # 缩放：仅对销量列基于训练集拟合
    scaler = MinMaxScalerPerColumn().fit(df_train, y_cols)

    # —— 统一一套特征列（从训练集/全量确定）——
    feat_cols_used = choose_feat_cols(df_feat)

    def arrays_with_feat_cols(df_part: pd.DataFrame):
        y_scaled = scaler.transform(df_part, y_cols)[y_cols].to_numpy()
        extra = df_part[feat_cols_used].to_numpy(dtype=float) if feat_cols_used else None
        return y_scaled, extra

    # 训练集
    y_tr, e_tr = arrays_with_feat_cols(df_train)
    X_tr, Y_tr = build_sliding_windows(y_tr, e_tr, seq_len, pred_len)
    train_ds = Seq2SeqSlidingDataset(X_tr, Y_tr)

    # 验证集（带上下文）
    y_va, e_va = arrays_with_feat_cols(df_val_ctx)
    X_va, Y_va = build_sliding_windows(y_va, e_va, seq_len, pred_len)
    if X_va.size == 0:
        raise ValueError(
            f"验证集滑窗为空：val_len={val_len}, seq_len={seq_len}, pred_len={pred_len}。"
            f"请增大 val_len 或减小 seq_len/pred_len。"
        )
    val_ds   = Seq2SeqSlidingDataset(X_va, Y_va)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, drop_last=False)

    # 测试输入：最后一个窗口
    last_input_start = max(0, len(df_feat) - test_len - seq_len)
    df_hist = df_feat.iloc[last_input_start:last_input_start+seq_len]
    y_hist, e_hist = arrays_with_feat_cols(df_hist)
    test_input = np.concatenate([y_hist, e_hist], axis=1) if e_hist is not None else y_hist
    test_input = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, F)

    # 真值（未缩放）用于指标
    y_true_future = df_feat[y_cols].iloc[-test_len:].to_numpy()[:pred_len, :]  # (pred_len, C)

    # 统计样本数
    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_test  = 1  # 我们在测试阶段仅使用 1 个窗口预测 pred_len 天

    return (train_loader, val_loader, scaler, test_input, y_true_future,
            y_cols, feat_cols_used, n_train, n_val, n_test)

def train_and_eval(cfg):
    device = pick_device(cfg.get('device','auto'))
    set_seed(cfg['seed'])

    # 1) 读取与缺失处理
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)

    # 2) 构建特征表
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    # 3) DataLoaders & 缩放器
    (train_loader, val_loader, scaler, test_input, y_true_future,
     y_cols, feat_cols_used, n_train, n_val, n_test) = make_loaders(feat_df, y_cols, cfg)

    # 输入维度
    input_size = len(y_cols) + len(feat_cols_used)

    # ===== 训练前信息打印 =====
    print("\n========== Training Setup ==========")
    print(f"Device                : {device}")
    print(f"Sequence length       : {cfg['seq_len']}")
    print(f"Prediction length     : {cfg['pred_len']}")
    print(f"Train samples         : {n_train}")
    print(f"Validation samples    : {n_val}")
    print(f"Test samples          : {n_test}")
    print(f"Series (counties)     : {len(y_cols)}")
    print(f"Extra features used   : {len(feat_cols_used)} -> {feat_cols_used}")
    print(f"Model input_size      : {input_size}")
    print("====================================\n")

    # 4) 建模
    model = LSTMSeq2Seq(
        input_size=input_size,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        pred_len=cfg['pred_len'],
        num_series=len(y_cols),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()

    # 5) 训练循环（早停）
    best_val = float('inf')
    best_state = None
    patience = cfg['patience']
    bad = 0

    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

    # 恢复最佳
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) 保存
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_root = cfg['outputs_dir']
    ckpt_dir = os.path.join(out_root, 'checkpoints')
    pred_dir = os.path.join(out_root, 'predictions')
    for d in [ckpt_dir, pred_dir]:
        os.makedirs(d, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save(model.state_dict(), ckpt_path)

    # 7) 测试阶段：最后窗口一次性预测
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(test_input.to(device)).cpu().numpy().reshape(cfg['pred_len'], len(y_cols))
    y_pred = scaler.inverse_transform_array(y_pred_scaled, y_cols)
    y_true = y_true_future[:cfg['pred_len'], :]

    # 指标
    metrics = per_series_metrics(y_true, y_pred, y_cols, eps=cfg.get('mape_eps', 1e-6))

    # 8) 保存预测结果和指标
    last_dates = feat_df.index[-cfg['test_len']:][:cfg['pred_len']]
    pred_df = pd.DataFrame(y_pred, columns=y_cols, index=last_dates)
    pred_csv = os.path.join(pred_dir, f'preds_{ts}.csv')
    pred_df.to_csv(pred_csv, encoding='utf-8-sig')

    met_json = os.path.join(pred_dir, f'metrics_{ts}.json')
    save_json(metrics, met_json)

    print('\n保存完成:')
    print('  模型  ->', ckpt_path)
    print('  预测  ->', pred_csv)
    print('  指标  ->', met_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    train_and_eval(cfg)
