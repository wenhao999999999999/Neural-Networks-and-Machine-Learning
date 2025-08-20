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

def plot_predictions(y_true, y_pred=None, y_future=None, dates=None, title="预测结果", label="值", save_path='prediction.png'):
    """
    绘制时间序列预测图，包含历史真实值、预测值和未来预测
    
    参数:
    - y_true: 历史真实值
    - y_pred: 历史预测值（可选）
    - y_future: 未来预测值（可选）
    - dates: 时间序列日期（用于x轴）
    - title: 图表标题
    - label: y轴标签
    - save_path: 保存路径
    """
    # 设置中文字体
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    if 'Microsoft YaHei' in font_list:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    elif 'SimHei' in font_list:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        print("⚠️ 未找到可用中文字体，中文可能无法正常显示")
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确定时间轴
    if dates is not None and len(dates) >= len(y_true):
        # 使用实际日期
        x_true = dates.iloc[:len(y_true)] if hasattr(dates, 'iloc') else dates[:len(y_true)]
        
        # 计算未来预测的时间轴（紧跟在历史数据之后）
        if y_future is not None and len(dates) > 0:
            last_date = dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1]
            future_x = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(y_future)
            )
    else:
        # 使用时间步作为x轴
        x_true = range(len(y_true))
        future_x = range(len(y_true), len(y_true) + len(y_future)) if y_future is not None else None
    
    # 绘制历史真实值
    plt.plot(x_true, y_true, label='历史真实值', color='blue', linewidth=2)
    
    # 绘制历史预测值（如果提供）
    if y_pred is not None:
        # 确保预测值长度与x轴匹配
        x_pred = x_true[:len(y_pred)] if len(y_pred) < len(x_true) else x_true
        plt.plot(x_pred, y_pred[:len(x_pred)], label='历史预测值', color='red', linestyle='--', linewidth=2)
    
    # 绘制未来预测值（如果提供）
    if y_future is not None and future_x is not None:
        plt.plot(future_x, y_future, label='未来预测值', color='green', linestyle='-.', linewidth=2)
    
    # 图表设置
    plt.title(title, fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # 如果使用日期，优化x轴显示
    if dates is not None:
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
