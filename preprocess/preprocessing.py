# -*- coding: utf-8 -*-
"""
功能汇总：
1) add_time_features: 与旧版一致的时间特征（is_holiday / weekday(_sin/_cos) / month(_sin/_cos) / is_month_end）
2) fill_missing_by_calendar: 兼容两种模式
   - 宽表多目标（group_col=None）：对每个系列列修复缺失（休息日=0，工作日插值）
   - 长表单目标（group_col 有值）：按分组对 target_col 修复
3) MinMaxScalerDict: 简洁列级缩放器（含保存/加载/反归一化）
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Union

import json
import numpy as np
import pandas as pd

# 优先使用 chinese_calendar；没有则周末视作休息日
try:
    import chinese_calendar as cncal
except Exception:
    cncal = None


# ------------ 时间特征 ------------
def _is_holiday_or_rest(day: pd.Timestamp) -> int:
    """1=节假日/休息日，0=工作日；无 chinese_calendar 时周末=休息日"""
    if cncal is not None:
        try:
            return int(not cncal.is_workday(day))
        except Exception:
            pass
    return int(day.weekday() >= 5)


def add_time_features(
    df: pd.DataFrame,
    date_col: str,
    add_time_features: bool = True,
    weekday_cyclical: bool = False,
    month_cyclical: bool = True,
) -> pd.DataFrame:
    if not add_time_features:
        return df.copy()

    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    out["is_holiday"]   = dt.map(_is_holiday_or_rest).astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)

    out["weekday"] = dt.dt.weekday
    if weekday_cyclical:
        out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
        out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)

    out["month"] = dt.dt.month
    if month_cyclical:
        out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)

    return out


# ------------ 缺失修复（宽表/长表） ------------
def fill_missing_by_calendar(
    df: pd.DataFrame,
    date_col: str,
    group_col: Optional[str],   # 宽表传 None
    target_col: Optional[str],  # 宽表传 None
    freq: str = "D",
) -> pd.DataFrame:
    """
    A) 宽表多目标（group_col=None）：
       - 按完整日期索引重建
       - 每个系列列：休息日缺失→0；工作日缺失→插值；最后 ffill/bfill 与裁负值
    B) 长表单目标（group_col 有值）：
       - 对每个分组的 target_col 做同样处理
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # === 宽表 ===
    if group_col is None:
        full_index = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
        dfx = df.set_index(date_col).reindex(full_index)
        dfx.index.name = date_col

        series_cols = [c for c in df.columns if c != date_col]
        for col in series_cols:
            if not pd.api.types.is_numeric_dtype(dfx[col]):  # 只处理数值列
                continue
            mask_nan = dfx[col].isna()
            if mask_nan.any():
                for day in dfx.index[mask_nan]:
                    if _is_holiday_or_rest(pd.Timestamp(day)):
                        dfx.loc[day, col] = 0.0
            dfx[col] = dfx[col].interpolate(method="linear", limit_direction="both")
            dfx[col] = dfx[col].ffill().bfill().fillna(0.0)
            dfx[col] = dfx[col].clip(lower=0.0)
        return dfx.reset_index()

    # === 长表 ===
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' 不在数据列中。")
    if target_col is None or target_col not in df.columns:
        raise KeyError("长表模式需要提供有效的 target_col（且该列存在于数据中）。")

    out_list = []
    for g, gdf in df.groupby(group_col):
        gdf = gdf.sort_values(date_col).reset_index(drop=True)
        full_index = pd.date_range(gdf[date_col].min(), gdf[date_col].max(), freq=freq)
        gdf_full = gdf.set_index(date_col).reindex(full_index)
        gdf_full.index.name = date_col

        mask_nan = gdf_full[target_col].isna()
        if mask_nan.any():
            for day in gdf_full.index[mask_nan]:
                if _is_holiday_or_rest(pd.Timestamp(day)):
                    gdf_full.loc[day, target_col] = 0.0

        gdf_full[target_col] = gdf_full[target_col].interpolate(method="linear", limit_direction="both")
        gdf_full[target_col] = gdf_full[target_col].ffill().bfill().fillna(0.0)
        gdf_full[target_col] = gdf_full[target_col].clip(lower=0.0)

        gdf_full[group_col] = g
        out_list.append(gdf_full.reset_index())

    return pd.concat(out_list, axis=0, ignore_index=True)


# ------------ 简洁列级缩放器 ------------
class MinMaxScalerDict:
    def __init__(self, feature_cols: List[str], clip: bool = True):
        self.feature_cols = list(feature_cols)
        self.clip = bool(clip)
        self.stats: Dict[str, Dict[str, float]] = {c: {"min": None, "max": None} for c in self.feature_cols}

    def fit(self, df: pd.DataFrame) -> "MinMaxScalerDict":
        for c in self.feature_cols:
            self.stats[c]["min"] = float(np.nanmin(df[c].values))
            self.stats[c]["max"] = float(np.nanmax(df[c].values))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.feature_cols:
            lo, hi = self.stats[c]["min"], self.stats[c]["max"]
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            v = (out[c] - lo) / denom
            if self.clip:
                v = v.clip(0.0, 1.0)
            out[c] = v
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.feature_cols:
            lo, hi = self.stats[c]["min"], self.stats[c]["max"]
            out[c] = out[c] * (hi - lo) + lo
        return out

    def save(self, path: Union[str, Path]):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"feature_cols": self.feature_cols, "stats": self.stats, "clip": self.clip},
                f, ensure_ascii=False, indent=2
            )

    @staticmethod
    def load(path: Union[str, Path]) -> "MinMaxScalerDict":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sc = MinMaxScalerDict(obj["feature_cols"], obj.get("clip", True))
        sc.stats = obj["stats"]
        return sc
