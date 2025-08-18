import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from datetime import datetime, timedelta

# 加载数据
result_dir = "./results/informer_custom_ftS_sl90_ll10_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"
pred = np.load(os.path.join(result_dir, "pred.npy"))         # (2816, 48, 7)
true = np.load(os.path.join(result_dir, "true.npy"))         # (2816, 48, 7)
real_pred = np.load(os.path.join(result_dir, "real_prediction.npy"))  # (1, 48, 7)

# 打印张量的形状
print("pred.shape:", pred.shape)
print("true.shape:", true.shape)
print("real_pred.shape:", real_pred.shape)

# 加载原始数据集获取归一化参数
data_path = './data/ELE/power_load.csv'  # 原始数据路径
df = pd.read_csv(data_path)

# 提取特征列
# features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
features = ['power_load']
data = df[features].values

# 计算归一化参数（使用训练集的均值和标准差）
train_size = int(len(data) * 0.7)  # 70% 训练集
train_data = data[:train_size]

# 计算每个特征的均值和标准差
means = train_data.mean(axis=0)
stds = train_data.std(axis=0)

# 避免除以零
stds[stds == 0] = 1

# 逆归一化函数
def inverse_transform(data, means, stds):
    return data * stds + means

# 对结果进行逆归一化
true_orig = inverse_transform(true, means, stds)
pred_orig = inverse_transform(pred, means, stds)
real_pred_orig = inverse_transform(real_pred, means, stds)

# 选择要可视化的样本和变量
sample_idx = pred.shape[0] - 1  # 最后一个样本（与real_pred对应）
var_idx = 0        # 第 n 个变量（根据需求调整）

# 提取原始尺度的数据
true_values = true_orig[sample_idx, :, var_idx]  # 真实值
pred_values = pred_orig[sample_idx, :, var_idx]  # 模型预测值
real_pred_values = real_pred_orig[0, :, var_idx]  # 预测的未来值


# 创建时间轴（ETTh1是每小时数据）
# 获取原始数据的时间戳
date_col = pd.to_datetime(df['date'])
last_date = date_col.iloc[-1]  # 最后一个数据点的时间

# 计算预测开始时间（最后一个样本对应的时间）
pred_start = last_date - timedelta(hours=48)  # 因为我们有48小时的预测

# 为不同序列创建时间点
pred_time_points = [pred_start + timedelta(hours=i) for i in range(pred.shape[1])]
real_pred_time_points = [pred_start + timedelta(hours=pred.shape[1]+i) for i in range(pred.shape[1])]

# 创建图表
plt.figure(figsize=(18, 8))

# 绘制真实值（蓝色） - 原始预测区间
plt.plot(pred_time_points, true_values, 'b-o', 
         markersize=5, linewidth=1.8, 
         label='Ground Truth (Prediction Period)', alpha=0.9)

# 绘制模型预测值（红色）
plt.plot(pred_time_points, pred_values, 'r--s', 
         markersize=5, linewidth=1.8, 
         label='Model Prediction', alpha=0.8)

# 绘制真实预测的未来值（绿色） - 接在预测值之后
plt.plot(real_pred_time_points, real_pred_values, 'g-.D', 
         markersize=6, linewidth=2.2, 
         label='Real Prediction (Future)', alpha=0.9)

# 添加连接线（从预测值结束到真实预测开始）
if len(pred_time_points) > 0 and len(real_pred_time_points) > 0:
    plt.plot([pred_time_points[-1], real_pred_time_points[0]], 
             [pred_values[-1], real_pred_values[0]], 
             'm--', linewidth=1.5, alpha=0.6, label='Prediction Transition')

# 添加误差阴影（真实值与模型预测的差异）
error = np.abs(true_values - pred_values)
plt.fill_between(pred_time_points, 
                 pred_values - error, 
                 pred_values + error, 
                 color='gray', alpha=0.15, 
                 label='Prediction Error')

# 标记关键时间点
plt.axvline(x=pred_time_points[0], color='k', linestyle='--', alpha=0.7, label='Prediction Start')
plt.axvline(x=pred_time_points[-1], color='purple', linestyle='--', alpha=0.7, label='Prediction End')
plt.axvline(x=real_pred_time_points[0], color='orange', linestyle='--', alpha=0.7, label='Future Prediction Start')

# 美化时间轴
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.gcf().autofmt_xdate()  # 自动旋转日期标签

# 添加标签和图例
plt.title(f'ETTh1 Prediction Comparison (Sample {sample_idx}, Variable {features[var_idx]})', fontsize=16)
plt.xlabel('Time', fontsize=13)
plt.ylabel('Value', fontsize=13)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)

# 添加统计信息框
mae_model = np.mean(np.abs(true_values - pred_values))
mae_future = np.mean(np.abs(true_values - real_pred_values))

stats_text = f"""Prediction Statistics (MAE):
Ground Truth vs Model: {mae_model:.4f}
Ground Truth vs Future: {mae_future:.4f}"""
plt.annotate(stats_text, 
             xy=(0.02, 0.95), 
             xycoords='axes fraction',
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))

# 添加时间范围标记
plt.annotate(f'Prediction Period: {pred_time_points[0].strftime("%Y-%m-%d %H:%M")} to {pred_time_points[-1].strftime("%Y-%m-%d %H:%M")}',
             xy=(0.5, 0.02), xycoords='axes fraction',
             ha='center', fontsize=10, bbox=dict(fc='white', alpha=0.7))

plt.annotate(f'Future Prediction: {real_pred_time_points[0].strftime("%Y-%m-%d %H:%M")} to {real_pred_time_points[-1].strftime("%Y-%m-%d %H:%M")}',
             xy=(0.5, 0.07), xycoords='axes fraction',
             ha='center', fontsize=10, bbox=dict(fc='white', alpha=0.7))

# 添加特征名称和归一化信息
feature_name = features[var_idx]
plt.annotate(f'Feature: {feature_name} | Original Scale', 
             xy=(0.02, 0.02), xycoords='axes fraction',
             fontsize=10, bbox=dict(fc='white', alpha=0.7))

plt.tight_layout()
plt.show()