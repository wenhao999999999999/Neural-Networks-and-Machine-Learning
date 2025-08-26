import pandas as pd
import numpy as np
import chinese_calendar as cc
from typing import List, Tuple, Dict

# -------------------- 日历与缺失值处理 --------------------
def _is_rest_day(d: pd.Timestamp) -> bool:
    """是否休息日：法定节假日 或 非调休的周末。"""
    return not cc.is_workday(d)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_holiday'] = df.index.map(lambda d: int(not cc.is_workday(d)))
    df['weekday'] = df.index.weekday  # 0-6
    df['month'] = df.index.month      # 1-12
    df['is_month_end'] = df.index.is_month_end.astype(int)
    return df

def fill_missing_by_rule(df: pd.DataFrame, y_cols: List[str]) -> pd.DataFrame:
    """
    休息日缺失→0；工作日缺失→插值；最后 ffill/bfill 兜底并裁负值。
    """
    df = df.copy()

    rest_mask = df.index.to_series().map(_is_rest_day).values
    for col in y_cols:
        mask = rest_mask & df[col].isna().values
        df.loc[mask, col] = 0.0

    for col in y_cols:
        df[col] = df[col].interpolate(method='time', limit_direction='both')
        df[col] = df[col].ffill().bfill().fillna(0.0)  # ← 修复 FutureWarning
        df[col] = df[col].clip(lower=0.0)

    return df

# -------------------- 读取与整理 --------------------
def load_and_prepare(
    csv_path: str,
    exclude_cols: List[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取CSV，返回：日期为索引的完整日频表 + 销量列名 y_cols。
    """
    exclude_cols = exclude_cols or []
    df = pd.read_csv(csv_path)

    date_col = None
    for c in df.columns:
        if c.lower() in ['date', 'dt', 'trade_dt']:
            date_col = c
            break
    if date_col is None:
        raise ValueError('未找到日期列，请将日期列命名为 date/dt/trade_dt 之一。')

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    y_cols = [c for c in df.columns if c not in [date_col] + exclude_cols]

    full_index = pd.date_range(df[date_col].min(), df[date_col].max(), freq='D')
    df = df.set_index(date_col).reindex(full_index)
    df.index.name = 'date'
    return df, y_cols

def build_feature_table(df: pd.DataFrame, y_cols: List[str],
                        add_time_features: bool = True,
                        weekday_cyclical: bool = True,
                        month_cyclical: bool = True) -> pd.DataFrame:
    df = df.copy()
    if add_time_features:
        df = add_calendar_features(df)
        if weekday_cyclical:
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
        if month_cyclical:
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    return df

# -------------------- 缩放器 --------------------
class MinMaxScalerPerColumn:
    def __init__(self):
        self.min_: Dict[str, float] = {}
        self.max_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]):
        for c in cols:
            self.min_[c] = float(df[c].min())
            self.max_[c] = float(df[c].max())
        return self

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            mn, mx = self.min_[c], self.max_[c]
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            out[c] = (out[c] - mn) / denom
        return out

    def inverse_transform_array(self, arr: np.ndarray, cols: List[str]) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, ...]
        out = arr.copy()
        for j, c in enumerate(cols):
            mn, mx = self.min_[c], self.max_[c]
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            out[..., j] = out[..., j] * denom + mn
        if out.shape[0] == 1:
            out = out[0]
        return out
