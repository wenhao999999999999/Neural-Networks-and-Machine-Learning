# -*- coding: utf-8 -*-
"""
同一张图上绘制：
- 测试集真值（合计）
- 测试集预测（合计）
- 未来预测（合计）

说明：
- 若 CSV 中包含反归一化列（y_true_inv_*/y_pred_inv_*、future 的 y_pred_inv_*），
  本脚本优先使用这些列；否则回退到归一化列（y_true_*/y_pred_*）。
- 输出图片路径由调用方传入（见 main_lstm.py 中的 plot_combined 调用）。
"""

from pathlib import Path
from typing import Union, List

import pandas as pd
import matplotlib.pyplot as plt

from tools.plot_utils import save_show


def _pick_cols(df: pd.DataFrame, base_prefix: str) -> List[str]:
    """
    在 df 中挑选列名：
    - 优先选择带有 *_inv_* 的反归一化列
    - 若没有 *_inv_*，则选择常规前缀列
    例如：base_prefix='y_true'，则优先 ['y_true_inv_xxx'...]，
         否则回退到 ['y_true_xxx'...]
    """
    inv_cols = [c for c in df.columns if c.startswith(base_prefix + "_inv_")]
    if len(inv_cols) > 0:
        return inv_cols
    return [c for c in df.columns if c.startswith(base_prefix + "_")]


def plot_combined(
    test_csv: Union[str, Path],
    future_csv: Union[str, Path],
    out_path: Union[str, Path]
):
    """
    参数
    ----
    test_csv : 测试集预测文件路径（evaluate_test 产出）
               需至少包含 'date' 列，以及 y_true_* / y_pred_*（或 *_inv_*）列
    future_csv : 未来预测文件路径（forecast_future 产出）
                 需至少包含 'date' 列，以及 y_pred_*（或 y_pred_inv_*）列
    out_path : 输出图片路径
    """
    test_csv = Path(test_csv)
    future_csv = Path(future_csv)

    if not test_csv.exists():
        raise FileNotFoundError(f"test_csv not found: {test_csv}")
    if not future_csv.exists():
        raise FileNotFoundError(f"future_csv not found: {future_csv}")

    # 读取 CSV（确保 date 为日期类型）
    df_test = pd.read_csv(test_csv, parse_dates=["date"])
    df_future = pd.read_csv(future_csv, parse_dates=["date"])

    # 选择列：优先 inv 列
    true_cols = _pick_cols(df_test, "y_true")
    pred_cols = _pick_cols(df_test, "y_pred")
    fut_cols = [c for c in df_future.columns if c.startswith("y_pred_inv_")]
    if len(fut_cols) == 0:
        fut_cols = [c for c in df_future.columns if c.startswith("y_pred_")]

    if len(true_cols) == 0 or len(pred_cols) == 0 or len(fut_cols) == 0:
        raise ValueError(
            "列名不完整：请确认 test_predictions.csv 中存在 y_true_/y_pred_（或 *_inv_）列，"
            "以及 future_forecast.csv 中存在 y_pred_（或 y_pred_inv_）列。"
        )

    # 按天聚合（合计）
    agg_true = df_test[["date"] + true_cols].set_index("date").sum(axis=1)
    agg_pred = df_test[["date"] + pred_cols].set_index("date").sum(axis=1)
    agg_future = df_future[["date"] + fut_cols].set_index("date").sum(axis=1)

    # 绘图
    fig = plt.figure(figsize=(12, 4.8))
    plt.plot(agg_true.index, agg_true.values, label="Test True (sum)")
    plt.plot(agg_pred.index, agg_pred.values, label="Test Pred (sum)")
    plt.plot(agg_future.index, agg_future.values, label="Future Forecast (sum)")

    plt.title("Combined — Sum over all series")
    plt.xlabel("Date")
    plt.ylabel("Sales (sum)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存
    save_show(fig, out_path)


def plot_per_series(
    test_csv: Union[str, Path],
    future_csv: Union[str, Path],
    out_dir: Union[str, Path]
):
    """为每个区县绘制单独图像"""
    test_csv = Path(test_csv)
    future_csv = Path(future_csv)
    out_dir = Path(out_dir)

    if not test_csv.exists():
        raise FileNotFoundError(f"test_csv not found: {test_csv}")
    if not future_csv.exists():
        raise FileNotFoundError(f"future_csv not found: {future_csv}")

    df_test = pd.read_csv(test_csv, parse_dates=["date"])
    df_future = pd.read_csv(future_csv, parse_dates=["date"])

    true_cols = _pick_cols(df_test, "y_true")
    pred_cols = _pick_cols(df_test, "y_pred")
    fut_cols = [c for c in df_future.columns if c.startswith("y_pred_")]

    if len(true_cols) != len(pred_cols) or len(pred_cols) != len(fut_cols):
        raise ValueError("列名数量不一致，无法逐区绘图")

    series_names = [c.split("_", 2)[-1] for c in true_cols]

    for t_col, p_col, f_col, name in zip(true_cols, pred_cols, fut_cols, series_names):
        fig = plt.figure(figsize=(12, 4.8))
        plt.plot(df_test["date"], df_test[t_col], label="Test True")
        plt.plot(df_test["date"], df_test[p_col], label="Test Pred")
        plt.plot(df_future["date"], df_future[f_col], label="Future Forecast")

        plt.title(name)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_show(fig, out_dir / f"{name}.png")
