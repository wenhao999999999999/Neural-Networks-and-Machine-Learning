import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter

from matplotlib import font_manager


def aggregate_hourly_to_daily(csv_path):
    """
    将小时级用电量数据聚合为日级数据
    参数:
        csv_path: CSV文件路径
    返回:
        daily_df: 按天聚合的DataFrame (date, daily_power_load)
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)

    # 2. 转换日期格式并设置为索引
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M')  # 解析原始时间格式
    df.set_index('date', inplace=True)  # 设置日期为索引

    # 3. 生成完整的小时时间索引，确保每一天都有24小时
    full_range = pd.date_range(start=df.index.min().floor('D'), end=df.index.max().ceil('D') - pd.Timedelta(hours=1), freq='h')
    df = df.reindex(full_range)

    # 4. 用插值法补齐缺失的小时数据（线性插值，若首尾缺失则用前后值填充）
    if 'power_load' in df.columns:
        df['power_load'] = df['power_load'].interpolate(method='linear', limit_direction='both')
    else:
        raise ValueError('CSV文件缺少power_load列')

    # 5. 按天重采样并求和
    daily_df = df.resample('D').sum()

    # 6. 重置索引并重命名列
    daily_df.reset_index(inplace=True)
    daily_df.rename(columns={'index': 'date', 'power_load': 'daily_power_load'}, inplace=True)

    return daily_df
def calculate_metrics(y_true, y_pred, scaler, args):
    """计算预测评价指标"""
    
    # 计算MSE, MAE, RMSE
    mse = np.mean((y_true - y_pred) **2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse


