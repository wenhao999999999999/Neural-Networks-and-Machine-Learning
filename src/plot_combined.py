import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from data_prep import load_and_prepare, fill_missing_by_rule, build_feature_table

def main(cfg_path, hist_pred_csv, future_pred_csv, save_dir, history_days=None, county=None):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读原始 -> 缺失处理 -> 特征（仅为取齐日历索引用）
    raw_df, y_cols = load_and_prepare(cfg['data_path'], cfg.get('exclude_cols', []))
    filled_df = fill_missing_by_rule(raw_df, y_cols)
    feat_df = build_feature_table(
        filled_df, y_cols,
        add_time_features=cfg.get('add_time_features', True),
        weekday_cyclical=cfg.get('weekday_cyclical', True),
        month_cyclical=cfg.get('month_cyclical', True),
    )

    # 2) 历史真实值窗口（末尾 N 天）
    if history_days is None:
        # 默认给足一点上下文：seq_len + test_len
        history_days = int(cfg['seq_len']) + int(cfg['test_len'])
    true_hist = feat_df[y_cols].iloc[-history_days:]  # (Hh, C)

    # 3) 测试集历史预测（preds_*.csv）
    hist_pred = pd.read_csv(hist_pred_csv, index_col=0, parse_dates=True)
    # 只保留目标县区列，避免列顺序错乱
    hist_pred = hist_pred.reindex(columns=y_cols)

    # 4) 未来预测（future_*.csv）
    future_pred = pd.read_csv(future_pred_csv, index_col=0, parse_dates=True)
    future_pred = future_pred.reindex(columns=y_cols)

    # 5) 画图：每个区县一张
    counties = [county] if county else y_cols
    last_date = feat_df.index[-1]

    for col in counties:
        plt.figure(figsize=(10, 4.5))

        # 历史真实
        plt.plot(true_hist.index, true_hist[col].values, label='Actual (history)')

        # 测试集历史预测（它的索引在最后 test_len 天中的前 pred_len 天）
        if col in hist_pred.columns:
            plt.plot(hist_pred.index, hist_pred[col].values, '--', label='Predicted (test)')

        # 未来预测（从数据最后一天+1开始）
        if col in future_pred.columns:
            plt.plot(future_pred.index, future_pred[col].values, ':', label='Predicted (future)')

        # 分界线：数据最后一天
        plt.axvline(x=last_date, linestyle='--', alpha=0.6)

        plt.title(f'{col} — Actual vs Test Pred vs Future Pred')
        plt.xlabel('Date'); plt.ylabel('Sales')
        plt.legend()
        plt.tight_layout()

        out = os.path.join(save_dir, f'{col}_combined.png')
        plt.savefig(out, bbox_inches='tight', dpi=150)
        plt.close()

    print('Saved combined plots ->', save_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--hist_pred_csv', type=str, required=True,
                    help='./outputs/predictions/preds_YYYYMMDD_HHMMSS.csv')
    ap.add_argument('--future_pred_csv', type=str, required=True,
                    help='./outputs/predictions/future_31d_YYYYMMDD_HHMMSS.csv')
    ap.add_argument('--save_dir', type=str, default='./outputs/plots_combined')
    ap.add_argument('--history_days', type=int, default=None,
                    help='历史真实值展示的天数窗口，默认 seq_len + test_len')
    ap.add_argument('--county', type=str, default=None,
                    help='只画指定列名（区县）；不填则为所有列')
    args = ap.parse_args()
    main(args.config, args.hist_pred_csv, args.future_pred_csv,
         args.save_dir, args.history_days, args.county)
