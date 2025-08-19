import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

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
    full_range = pd.date_range(start=df.index.min().floor('D'), end=df.index.max().ceil('D') - pd.Timedelta(hours=1), freq='H')
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
    """计算 MSE、MAE、RMSE（支持反归一化）"""
    # 转换为 numpy 数组
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 反归一化（还原真实尺度）
    if args.features == 'S':
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true = scaler.inverse_transform(y_true.reshape(-1, args.input_dim)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, args.input_dim)).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae, np.sqrt(mse)

def plot_predictions(y_true, y_pred, scaler, args, save_path='img/prediction.png'):
    """绘制预测曲线（支持单/多变量）"""
    # 反归一化
    if args.features == 'S':
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        label = args.target_col
    else:
        y_true = scaler.inverse_transform(y_true.reshape(-1, args.input_dim)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, args.input_dim)).flatten()
        label = args.data_name  # 多变量时用数据集名
    
    # 绘制前 200 个点（避免图拥挤）
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:200], label=f'Real {label}', color='blue')
    plt.plot(y_pred[:200], label=f'Predicted {label}', color='red', linestyle='--')
    plt.title(f'LSTM Prediction (seq_len={args.seq_len}, pred_len={args.pred_len})')
    plt.xlabel('Time Step')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


