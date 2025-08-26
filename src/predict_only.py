import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from utils import pick_device
from model import LSTMSeq2Seq
from data_prep import load_and_prepare, fill_missing_by_rule, build_feature_table, MinMaxScalerPerColumn

@torch.no_grad()
def run_predict(cfg, ckpt_path):
    device = pick_device(cfg.get('device','auto'))

    # 读取和特征
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    # 缩放器 (基于训练段拟合)
    T = len(feat_df)
    train_end = T - cfg['val_len'] - cfg['test_len']
    scaler = MinMaxScalerPerColumn().fit(feat_df.iloc[:train_end], y_cols)

    # 准备最后窗口输入
    seq_len = cfg['seq_len']; pred_len = cfg['pred_len']
    last_input_start = max(0, len(feat_df) - cfg['test_len'] - seq_len)
    df_hist = feat_df.iloc[last_input_start:last_input_start+seq_len]

    y_hist = scaler.transform(df_hist, y_cols)[y_cols].to_numpy()
    feat_cols = [c for c in df_hist.columns if c not in y_cols]
    extra_hist = df_hist[feat_cols].to_numpy(dtype=float) if len(feat_cols)>0 else None
    x = np.concatenate([y_hist, extra_hist], axis=1) if extra_hist is not None else y_hist
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # 模型
    input_size = x.shape[-1]
    model = LSTMSeq2Seq(input_size, cfg['hidden_dim'], cfg['num_layers'], cfg['dropout'], pred_len, len(y_cols)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_pred_scaled = model(x).cpu().numpy().reshape(pred_len, len(y_cols))
    y_pred = scaler.inverse_transform_array(y_pred_scaled, y_cols)

    # 保存
    last_dates = feat_df.index[-cfg['test_len']:][:pred_len]
    pred_df = pd.DataFrame(y_pred, columns=y_cols, index=last_dates)

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    pred_dir = os.path.join(cfg['outputs_dir'], 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    pred_csv = os.path.join(pred_dir, f'preds_only_{ts}.csv')
    pred_df.to_csv(pred_csv, encoding='utf-8-sig')
    print('预测保存 ->', pred_csv)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    run_predict(cfg, args.ckpt)
