import os, argparse
import numpy as np
import pandas as pd
import torch
import yaml

from model import LSTMSeq2Seq
from utils import pick_device
from data_prep import load_and_prepare, fill_missing_by_rule, build_feature_table, MinMaxScalerPerColumn

def choose_feat_cols(df_part: pd.DataFrame):
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

@torch.no_grad()
def main(cfg_path, ckpt_path, days=None):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    device = pick_device(cfg.get('device','auto'))

    # 读取 & 特征
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    pred_len = cfg['pred_len'] if days is None else int(days)
    seq_len = cfg['seq_len']

    # 缩放器基于训练段拟合
    T = len(feat_df)
    train_end = T - cfg['val_len'] - cfg['test_len']
    scaler = MinMaxScalerPerColumn().fit(feat_df.iloc[:train_end], y_cols)

    # 准备最后窗口输入
    feat_cols_used = choose_feat_cols(feat_df)
    df_hist = feat_df.iloc[-seq_len:]
    y_hist = scaler.transform(df_hist, y_cols)[y_cols].to_numpy()
    extra_hist = df_hist[feat_cols_used].to_numpy(dtype=float) if feat_cols_used else None
    x = np.concatenate([y_hist, extra_hist], axis=1) if extra_hist is not None else y_hist
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # 模型
    input_size = x.shape[-1]
    model = LSTMSeq2Seq(
        input_size=input_size,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        pred_len=cfg['pred_len'],
        num_series=len(y_cols),
    ).to(device)
    
    # 安全加载 state_dict（支持新版本的 weights_only 参数，老版本自动回退）
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    model.eval()

    # 推理（若 days != cfg['pred_len']，做截断或报错）
    y_pred_scaled = model(x).cpu().numpy().reshape(cfg['pred_len'], len(y_cols))
    if pred_len < cfg['pred_len']:
        y_pred_scaled = y_pred_scaled[:pred_len]
    elif pred_len > cfg['pred_len']:
        raise ValueError(f"当前模型固定预测步数为 {cfg['pred_len']}，想预测 {pred_len} 天请修改 config 或重新训练。")

    y_pred = scaler.inverse_transform_array(y_pred_scaled, y_cols)

    # 未来日期索引：从数据最后一天 + 1 开始
    start_next = feat_df.index[-1] + pd.Timedelta(days=1)
    future_index = pd.date_range(start_next, periods=pred_len, freq='D')
    pred_df = pd.DataFrame(y_pred, columns=y_cols, index=future_index)

    # 保存
    out_dir = os.path.join(cfg['outputs_dir'], 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(out_dir, f'future_{pred_len}d_{ts}.csv')
    pred_df.to_csv(out_csv, encoding='utf-8-sig')
    print('未来预测保存 ->', out_csv)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--days', type=int, default=None, help='未来预测天数，缺省用 config.pred_len')
    args = ap.parse_args()
    main(args.config, args.ckpt, args.days)
