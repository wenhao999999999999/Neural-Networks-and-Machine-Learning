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


# ---------------------- feature helpers ----------------------
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


def _make_future_time_feats(start_date: pd.Timestamp, days: int, feat_cols_used):
    """Generate minimal future time features aligned with training columns.
    If 'is_holiday' is required, we default to 0.0 (workdays). Plug a holiday lib if needed."""
    idx = pd.date_range(start_date + pd.Timedelta(days=1), periods=days, freq='D')
    feats = pd.DataFrame(index=idx)

    if 'weekday' in feat_cols_used:
        feats['weekday'] = feats.index.weekday.astype(float)
    if 'weekday_sin' in feat_cols_used and 'weekday_cos' in feat_cols_used:
        w = feats.index.weekday.values
        feats['weekday_sin'] = np.sin(2*np.pi*(w/7.0))
        feats['weekday_cos'] = np.cos(2*np.pi*(w/7.0))

    if 'month' in feat_cols_used:
        feats['month'] = feats.index.month.astype(float)
    if 'month_sin' in feat_cols_used and 'month_cos' in feat_cols_used:
        m = feats.index.month.values
        feats['month_sin'] = np.sin(2*np.pi*(m/12.0))
        feats['month_cos'] = np.cos(2*np.pi*(m/12.0))

    if 'is_month_end' in feat_cols_used:
        feats['is_month_end'] = feats.index.is_month_end.astype(float)
    if 'is_holiday' in feat_cols_used:
        feats['is_holiday'] = 0.0

    for c in feat_cols_used:
        if c not in feats.columns:
            feats[c] = 0.0
    return feats[feat_cols_used]


# ---------------------- split & loaders ----------------------
def compute_split_lengths(T: int, cfg: dict):
    pred_len = int(cfg['pred_len'])
    use_ratio = bool(cfg.get('split_by_ratio', False)) or \
                ('val_ratio' in cfg or 'test_ratio' in cfg or 'train_ratio' in cfg)
    info = {}

    if use_ratio:
        tr = cfg.get('train_ratio', None)
        vr = cfg.get('val_ratio', 0.2)
        te = cfg.get('test_ratio', 0.1)
        if tr is None:
            tr = max(0.0, 1.0 - float(vr) - float(te))
        tr, vr, te = float(tr), float(vr), float(te)
        s = tr + vr + te
        if s <= 0:
            raise ValueError("train/val/test 比例之和必须 > 0。")
        tr, vr, te = tr/s, vr/s, te/s

        train_len = int(round(T * tr))
        val_len   = int(round(T * vr))
        test_len  = int(T - train_len - val_len)
        train_len = max(train_len, 1)
        val_len   = max(val_len,   1)
        test_len  = max(test_len,  1)

        if val_len < pred_len:
            need = pred_len - val_len
            val_len = pred_len
            if train_len > need:
                train_len -= need
            elif train_len + test_len > need:
                borrow = need - train_len + 1
                test_len = max(1, test_len - borrow)
                train_len = 1
            else:
                raise ValueError("数据太短，无法满足 val_len >= pred_len 的要求。")

        info.update({"mode": "ratio", "train_ratio": tr, "val_ratio": vr, "test_ratio": te})
    else:
        val_len  = int(cfg.get('val_len', 31))
        test_len = int(cfg.get('test_len', 31))
        train_len = int(T - val_len - test_len)
        if min(train_len, val_len, test_len) <= 0:
            raise ValueError(f"切分无效：T={T}, train_len={train_len}, val_len={val_len}, test_len={test_len}")
        if val_len < pred_len:
            raise ValueError(f"验证集天数 val_len({val_len}) 必须 >= pred_len({pred_len})。")
        info.update({"mode": "length"})

    return train_len, val_len, test_len, info


def make_loaders(df_feat: pd.DataFrame, y_cols, cfg):
    seq_len = int(cfg['seq_len'])
    pred_len = int(cfg['pred_len'])

    T = len(df_feat)
    train_len, val_len, test_len, split_info = compute_split_lengths(T, cfg)
    assert val_len >= pred_len

    train_end = train_len
    val_end   = train_len + val_len

    df_train = df_feat.iloc[:train_end]
    start_val_ctx = max(0, train_end - seq_len)
    df_val_ctx = df_feat.iloc[start_val_ctx:val_end]

    scaler = MinMaxScalerPerColumn().fit(df_train, y_cols)
    feat_cols_used = choose_feat_cols(df_feat)

    def arrays_with_feat_cols(df_part: pd.DataFrame):
        y_scaled = scaler.transform(df_part, y_cols)[y_cols].to_numpy()
        extra = df_part[feat_cols_used].to_numpy(dtype=float) if feat_cols_used else None
        return y_scaled, extra

    # 训练
    y_tr, e_tr = arrays_with_feat_cols(df_train)
    X_tr, Y_tr = build_sliding_windows(y_tr, e_tr, seq_len, pred_len)
    train_ds = Seq2SeqSlidingDataset(X_tr, Y_tr)

    # 验证（带上下文）
    y_va, e_va = arrays_with_feat_cols(df_val_ctx)
    X_va, Y_va = build_sliding_windows(y_va, e_va, seq_len, pred_len)
    if X_va.size == 0:
        raise ValueError(
            f"验证集滑窗为空：val_len={val_len}, seq_len={seq_len}, pred_len={pred_len}。"
            f"请增大 val_len 或减小 seq_len/pred_len。"
        )
    val_ds   = Seq2SeqSlidingDataset(X_va, Y_va)

    # 测试逐滑窗（从 val_end - seq_len 开始建立上下文）
    start_test_ctx = max(0, val_end - seq_len)
    df_test_ctx = df_feat.iloc[start_test_ctx:]
    y_te, e_te = arrays_with_feat_cols(df_test_ctx)
    X_te, Y_te = build_sliding_windows(y_te, e_te, seq_len, pred_len)
    test_ds = Seq2SeqSlidingDataset(X_te, Y_te)

    train_loader = DataLoader(train_ds, batch_size=int(cfg['batch_size']), shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=int(cfg['batch_size']), shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=int(cfg.get('eval_batch_size', cfg['batch_size'])), shuffle=False, drop_last=False)

    split_stats = {
        "T": T, "train_len": train_len, "val_len": val_len, "test_len": test_len,
        "split_info": split_info, "n_test_windows": len(test_ds)
    }

    return (train_loader, val_loader, test_loader, scaler,
            y_cols, feat_cols_used, len(train_ds), len(val_ds), len(test_ds), df_feat, split_stats)


# ---------------------- train & eval ----------------------
def train_and_eval(cfg, run_name: str = None, save_outputs: bool = True):
    device = pick_device(cfg.get('device','auto'))
    set_seed(cfg.get('seed', 42))

    # 1) 读取+缺失处理
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)

    # 2) 特征表
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    # 3) Loaders
    (train_loader, val_loader, test_loader, scaler,
     y_cols, feat_cols_used, n_train, n_val, n_test,
     feat_df_full, split_stats) = make_loaders(feat_df, y_cols, cfg)

    input_size = len(y_cols) + len(feat_cols_used)

    print("\n========== Training Setup ==========")
    print(f"Device                : {device}")
    print(f"Sequence length       : {cfg['seq_len']}")
    print(f"Prediction length     : {cfg['pred_len']}")
    if split_stats['split_info'].get('mode') == 'ratio':
        tr = split_stats['split_info']['train_ratio']
        vr = split_stats['split_info']['val_ratio']
        te = split_stats['split_info']['test_ratio']
        print(f"Split mode            : ratio (train/val/test = {tr:.3f}/{vr:.3f}/{te:.3f})")
    else:
        print("Split mode            : length (use val_len/test_len)")
    print(f"Split days (T={split_stats['T']}) : train={split_stats['train_len']} | val={split_stats['val_len']} | test={split_stats['test_len']}")
    print(f"Train samples         : {n_train}")
    print(f"Validation samples    : {n_val}")
    print(f"Test windows          : {n_test}  (逐滑窗评估)")
    print(f"Series (counties)     : {len(y_cols)}")
    print(f"Extra features used   : {len(feat_cols_used)} -> {feat_cols_used}")
    print(f"Model input_size      : {input_size}")
    print("====================================\n")

    # 4) 模型
    model = LSTMSeq2Seq(
        input_size=input_size,
        hidden_dim=int(cfg['hidden_dim']),
        num_layers=int(cfg['num_layers']),
        dropout=float(cfg['dropout']),
        pred_len=int(cfg['pred_len']),
        num_series=len(y_cols),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']))
    criterion = nn.MSELoss()

    # 5) 训练（早停）
    best_val = float('inf')
    best_state = None
    patience = int(cfg['patience'])
    bad = 0

    for epoch in range(1, int(cfg['num_epochs']) + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) 全测试集逐滑窗评估
    model.eval()
    preds_scaled_list, trues_scaled_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred_scaled = model(xb).cpu().numpy()  # could be (B, C) or (B, pred_len, C)
            preds_scaled_list.append(pred_scaled)
            trues_scaled_list.append(yb.numpy())

    # unify shapes to (N, C)
    preds_scaled_cat = np.concatenate(preds_scaled_list, axis=0)
    if preds_scaled_cat.ndim == 3:
        # (N, pred_len, C) -> use only first step to align with yb which is usually single-step per window
        preds_scaled = preds_scaled_cat[:, 0, :]
    else:
        preds_scaled = preds_scaled_cat  # (N, C)

    trues_scaled = np.concatenate(trues_scaled_list, axis=0)
    if trues_scaled.ndim == 3:
        trues_scaled = trues_scaled[:, 0, :]

    preds = scaler.inverse_transform_array(preds_scaled, y_cols)
    trues = scaler.inverse_transform_array(trues_scaled, y_cols)

    test_avg_metrics = per_series_metrics(trues, preds, y_cols, eps=float(cfg.get('mape_eps', 1e-6)))
    print(f"[TEST-AVG] windows={len(preds)} | MAE={test_avg_metrics['OVERALL']['MAE']:.6f} | "
          f"RMSE={test_avg_metrics['OVERALL']['RMSE']:.6f} | MAPE={test_avg_metrics['OVERALL']['MAPE']:.6f}")

    # 7) 未来一个月滚动预测（preds_*.csv 改为未来 N 天；默认 31）
    future_days = int(cfg.get('future_days', 31))

    seq_len = int(cfg['seq_len'])
    feat_cols_used_hist = choose_feat_cols(feat_df)

    # 历史窗口：最后 seq_len 天
    last_hist_df = feat_df_full.iloc[-seq_len:]
    scaler_all = MinMaxScalerPerColumn().fit(feat_df_full, y_cols)

    y_hist_scaled = scaler_all.transform(last_hist_df, y_cols)[y_cols].to_numpy()  # (seq_len, C)
    e_hist = last_hist_df[feat_cols_used_hist].to_numpy(dtype=float) if feat_cols_used_hist else None

    # 未来时间特征
    future_feats = _make_future_time_feats(feat_df_full.index[-1], future_days, feat_cols_used_hist)

    # 递推
    y_history = y_hist_scaled.copy()
    e_history = e_hist.copy() if e_hist is not None else None
    preds_future_scaled = []

    for i in range(future_days):
        # 组装当前输入 (seq_len, C [+ E])
        if e_history is not None:
            cur_in = np.concatenate([y_history[-seq_len:], e_history[-seq_len:]], axis=1)
        else:
            cur_in = y_history[-seq_len:]
        xb = torch.tensor(cur_in, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out_scaled = model(xb).cpu().numpy()  # (1, C) or (1, pred_len, C)

        # 兼容输出形状，统一取第一个时间步
        if out_scaled.ndim == 3:
            step_scaled_vec = out_scaled[0, 0, :]  # (C,)
        elif out_scaled.ndim == 2:
            step_scaled_vec = out_scaled[0, :]     # (C,)
        else:
            raise ValueError(f"Unexpected output shape from model: {out_scaled.shape}")

        preds_future_scaled.append(step_scaled_vec.reshape(1, -1))

        # 更新历史（追加一步预测 & 对应时间特征）
        y_history = np.vstack([y_history, step_scaled_vec.reshape(1, -1)])
        if e_history is not None:
            e_next = future_feats.iloc[i].to_numpy(dtype=float).reshape(1, -1)
            e_history = np.vstack([e_history, e_next])

    preds_future_scaled = np.vstack(preds_future_scaled)  # (future_days, C)
    preds_future = scaler_all.inverse_transform_array(preds_future_scaled, y_cols)

    # 指标打包（TEST_AVG + 未来预测与可比真值的对齐片段）
    # 注意：未来预测没有真值，这里仅保留 TEST_AVG；LAST_WINDOW 可选地与测试集首段对齐评估
    out_metrics = {"TEST_AVG": test_avg_metrics}

    result = {
        "metrics": out_metrics,
        "best_val": float(best_val),
        "run_name": run_name or ""
    }

    # 8) 保存产物
    if save_outputs:
        ts = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(cfg['outputs_dir'], 'runs', run_name) if run_name else cfg['outputs_dir']
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        pred_dir = os.path.join(run_dir, 'predictions')
        for d in [ckpt_dir, pred_dir]:
            os.makedirs(d, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
        torch.save(model.state_dict(), ckpt_path)

        # 测试集逐滑窗预测（便于复查）
        test_avg_csv = os.path.join(pred_dir, f'test_windows_pred_{ts}.csv')
        pd.DataFrame(preds, columns=y_cols).to_csv(test_avg_csv, index=False, encoding='utf-8-sig')

        # 未来 N 天预测 CSV（index 从最后一天+1 开始）
        future_index = pd.date_range(feat_df_full.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
        pred_df_future = pd.DataFrame(preds_future, columns=y_cols, index=future_index)

        pred_csv_future = os.path.join(pred_dir, f'preds_{ts}.csv')
        pred_df_future.to_csv(pred_csv_future, encoding='utf-8-sig', index_label="date")


        met_json = os.path.join(pred_dir, f'metrics_{ts}.json')
        save_json(out_metrics, met_json)

        print('\n保存完成:')
        print('  模型  ->', ckpt_path)
        print('  逐滑窗预测CSV ->', test_avg_csv)
        print('  未来预测CSV   ->', pred_csv_future)
        print('  指标(JSON)   ->', met_json)

        result.update({
            "pred_csv": pred_csv_future,
            "ckpt_path": ckpt_path,
            "run_dir": run_dir,
        })

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_save_outputs', action='store_true', help='若设置，则不保存模型与预测文件')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    _ = train_and_eval(cfg, run_name=args.run_name, save_outputs=not args.no_save_outputs)