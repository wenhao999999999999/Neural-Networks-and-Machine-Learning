import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import font_manager

# 设置中文字体
font_list = [f.name for f in font_manager.fontManager.ttflist]
if 'Microsoft YaHei' in font_list:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif 'SimHei' in font_list:
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    print("⚠️ 未找到可用中文字体，中文可能无法正常显示")
plt.rcParams['axes.unicode_minus'] = False

# 导入项目中的相关模块
from models.lstm_model import LSTMForecaster
from utils.tools import aggregate_hourly_to_daily

def load_model_and_scaler(model_path, seq_len, pred_len, hidden_dim, num_layers, dropout=0.2):
    """加载训练好的模型和归一化器"""
    # 创建模型参数命名空间
    class Args:
        def __init__(self):
            self.features = 'S'
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.data_dir = './data/'
            self.data_subdir = 'ELE'
            self.data_file = 'power_load.csv'
            self.data_name = 'ELE'
            self.target_col = 'daily_power_load'
            self.use_gpu = True if torch.cuda.is_available() else False
            self.gpu = 0

    args = Args()
    
    # 构建模型
    model = LSTMForecaster(args)
    
    # 加载模型权重
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 准备数据以获取归一化器
    data_path = f"{args.data_dir}/{args.data_subdir}/{args.data_file}"
    df = aggregate_hourly_to_daily(data_path)
    data = df[[args.target_col]].values
    
    # 划分训练集以拟合归一化器（与训练时保持一致）
    total_len = len(data)
    test_len = min(max(seq_len + pred_len, int(total_len * 0.15)), total_len - 2 * (seq_len + pred_len))
    val_len = min(max(seq_len + pred_len, int(total_len * 0.15)), total_len - test_len - (seq_len + pred_len))
    train_len = total_len - val_len - test_len
    train_data = data[:train_len]
    
    # 拟合归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    return model, scaler, args, device, df

def rolling_forecast(model, scaler, args, device, df, day_len):
    """
    滚动预测未来day_len天的电力负荷
    
    参数:
        model: 训练好的LSTM模型
        scaler: 归一化器
        args: 模型参数
        device: 运行设备
        df: 原始数据DataFrame
        day_len: 要预测的总天数
        
    返回:
        future_predictions: 预测结果（反归一化后）
        future_dates: 预测结果对应的日期
        last_known_date: 已知数据的最后日期
    """
    # 获取最后seq_len个已知数据作为初始输入
    target_col = args.target_col
    data = df[[target_col]].values
    last_sequence = data[-args.seq_len:]
    
    # 归一化初始序列
    last_sequence_norm = scaler.transform(last_sequence)
    
    # 计算需要进行多少次预测
    pred_len = args.pred_len
    num_pred_steps = (day_len + pred_len - 1) // pred_len  # 向上取整
    
    # 存储所有预测结果
    all_predictions = []
    
    # 滚动预测
    current_sequence = last_sequence_norm.copy()
    
    for i in range(num_pred_steps):
        # 准备输入
        input_seq = torch.tensor(
            current_sequence.reshape(1, args.seq_len, -1),
            dtype=torch.float32
        ).to(device)
        
        # 预测
        with torch.no_grad():
            pred = model(input_seq).cpu().numpy()[0]
        
        # 反归一化预测结果（用于存储）
        pred_denorm = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        
        # 保存预测结果（注意最后一步可能需要截断）
        remaining_days = day_len - len(all_predictions)
        if remaining_days < pred_len:
            all_predictions.extend(pred_denorm[:remaining_days])
        else:
            all_predictions.extend(pred_denorm)
        
        # 更新输入序列：后移pred_len，并添加预测结果
        # 归一化预测结果用于更新序列
        pred_norm = pred.reshape(-1, 1)
        # 移除前pred_len个数据，添加新的预测结果
        current_sequence = np.concatenate([current_sequence[pred_len:], pred_norm])
    
    # 生成预测日期
    last_known_date = pd.to_datetime(df['date'].iloc[-1])
    future_dates = [last_known_date + timedelta(days=i+1) for i in range(day_len)]
    
    return np.array(all_predictions), future_dates, last_known_date

def plot_predictions(df, future_predictions, future_dates, last_known_date, args):
    """绘制真实值和预测值"""
    plt.figure(figsize=(16, 8))
    
    # 绘制已知数据（最后3个月用于参考）
    plot_start_date = last_known_date - timedelta(days=90)  # 显示最后90天的已知数据
    plot_df = df[df['date'] >= plot_start_date]
    
    plt.plot(plot_df['date'], plot_df[args.target_col], 'b-', label='已知真实值', linewidth=2)
    
    # 绘制预测数据
    plt.plot(future_dates, future_predictions, 'r--', label='预测值', linewidth=2)
    
    # 添加分割线
    plt.axvline(x=last_known_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    plt.text(last_known_date, plt.ylim()[1]*0.95, '预测开始', 
             rotation=90, verticalalignment='top', fontsize=10)
    
    plt.title(f'电力负荷滚动预测 (总预测天数: {len(future_predictions)})', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('每日电力负荷', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图像
    result_dir = os.path.dirname(model_path)
    img_dir = os.path.join(result_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f'rolling_prediction_{len(future_predictions)}days.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"预测图像已保存至: {save_path}")

def predict_future(day_len, model_path):
    """
    预测未来指定天数的电力负荷并绘图
    
    参数:
        day_len: 要预测的天数
        model_path: 模型权重文件路径
    """
    # 模型参数（从路径和配置中获取）
    seq_len = 60
    pred_len = 7
    hidden_dim = 64
    num_layers = 2
    
    # 加载模型、归一化器和数据
    model, scaler, args, device, df = load_model_and_scaler(
        model_path, seq_len, pred_len, hidden_dim, num_layers
    )
    
    # 执行滚动预测
    print(f"开始滚动预测未来 {day_len} 天的电力负荷...")
    future_predictions, future_dates, last_known_date = rolling_forecast(
        model, scaler, args, device, df, day_len
    )
    
    # 绘制结果
    plot_predictions(df, future_predictions, future_dates, last_known_date, args)
    
    return future_predictions, future_dates

if __name__ == "__main__":
    # 模型路径
    model_path = "results/lstm_ELE_S_sl60_pl7_hd64_nl2_0/lstm_ELE_S_sl60_pl7_hd64_nl2_0_best.pth"
    
    # 预测未来30天（可根据需要修改）
    predict_days = 30
    predictions, dates = predict_future(predict_days, model_path)
    
    # 打印部分预测结果
    print("\n预测结果（前10天）:")
    for date, pred in zip(dates[:10], predictions[:10]):
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f}")