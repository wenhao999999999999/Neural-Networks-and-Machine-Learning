import os, argparse, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from data_prep import load_and_prepare, fill_missing_by_rule, build_feature_table

def main(cfg_path, pred_csv, save_dir):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读真实数据（用于取最后 test_len 天里的前 pred_len 天当作 “真值”）
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    pred_len = cfg['pred_len']; test_len = cfg['test_len']
    true_df = feat_df[y_cols].iloc[-test_len:][:pred_len]  # (pred_len, C)

    # 2) 读预测
    pred_df = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
    pred_df = pred_df[y_cols]  # 保证列顺序一致
    # 对齐索引（若与你保存的索引一致则无需这步）
    pred_df = pred_df.iloc[:pred_len]

    # 3) 逐区县画图
    for col in y_cols:
        plt.figure()
        plt.plot(true_df.index, true_df[col].values, label='Actual')
        plt.plot(pred_df.index, pred_df[col].values, '--', label='Predicted')
        plt.title(f'{col} - Actual vs Predicted')
        plt.xlabel('Date'); plt.ylabel('Sales')
        plt.legend()
        out = os.path.join(save_dir, f'{col}_actual_vs_pred.png')
        plt.savefig(out, bbox_inches='tight'); plt.close()

    # 4) 逐步(horizon)误差曲线（跨区县平均）
    y_true = true_df.to_numpy()      # (H, C)
    y_pred = pred_df.to_numpy()      # (H, C)
    mae = np.mean(np.abs(y_pred - y_true), axis=1)  # (H,)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(y_true), 1e-6)
        mape = np.mean(np.abs((y_pred - y_true) / denom), axis=1) * 100.0

    # MAE
    plt.figure()
    plt.plot(np.arange(1, pred_len+1), mae)
    plt.title('Horizon-wise MAE (avg over counties)')
    plt.xlabel('Forecast step (day)'); plt.ylabel('MAE')
    out = os.path.join(save_dir, 'horizon_mae.png')
    plt.savefig(out, bbox_inches='tight'); plt.close()

    # MAPE
    plt.figure()
    plt.plot(np.arange(1, pred_len+1), mape)
    plt.title('Horizon-wise MAPE% (avg over counties)')
    plt.xlabel('Forecast step (day)'); plt.ylabel('MAPE %')
    out = os.path.join(save_dir, 'horizon_mape.png')
    plt.savefig(out, bbox_inches='tight'); plt.close()

    # 5) 另存一个简单表格（首尾几天）
    preview = pd.DataFrame({
        'date': pred_df.index,
        **{f'{c}_true': true_df[c].values for c in y_cols},
        **{f'{c}_pred': pred_df[c].values for c in y_cols},
    })
    preview_path = os.path.join(save_dir, 'preview_true_vs_pred.csv')
    preview.to_csv(preview_path, index=False, encoding='utf-8-sig')
    print('Saved plots to:', save_dir)
    print('Preview CSV ->', preview_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--pred_csv', type=str, required=True,
                    help='路径形如 ./outputs/predictions/preds_YYYYMMDD_HHMMSS.csv')
    ap.add_argument('--save_dir', type=str, default='./outputs/plots')
    args = ap.parse_args()
    main(args.config, args.pred_csv, args.save_dir)
