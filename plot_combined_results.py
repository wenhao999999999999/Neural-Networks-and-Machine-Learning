import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta
from utils.tools import aggregate_hourly_to_daily
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

def print_npy_shapes(folder_path):
    """
    打印指定文件夹中所有.npy文件的形状
    
    参数:
        folder_path: 包含.npy文件的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
        
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以.npy结尾
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 加载.npy文件
                data = np.load(file_path)
                # 获取并打印形状
                print(f"{filename}形状: {data.shape}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

def get_test_dates(args, sample_count):
    """获取测试集样本对应的日期范围"""
    # 加载原始数据（日级聚合后的数据）
    data_path = f"{args.data_dir}/{args.data_subdir}/{args.data_file}"
    raw_df = aggregate_hourly_to_daily(data_path)
    raw_dates = raw_df['date']  # 完整日期序列（datetime格式）
    
    # 计算测试集长度
    total_len = len(raw_df)
    test_len = sample_count + args.seq_len + args.pred_len - 1
    
    # 测试集在原始数据中的起始索引
    test_start_idx = total_len - test_len
    
    # 测试集样本覆盖的日期范围
    first_pred_start = test_start_idx + args.seq_len
    last_pred_end = test_start_idx + test_len
    
    # 确保索引在有效范围内
    if first_pred_start >= len(raw_dates):
        first_pred_start = len(raw_dates) - 1
    if last_pred_end > len(raw_dates):
        last_pred_end = len(raw_dates)
    
    test_dates = raw_dates.iloc[first_pred_start:last_pred_end]
    
    return test_dates

def generate_future_dates(last_test_date, future_steps):
    """生成未来预测的日期序列"""
    future_dates = []
    current_date = last_test_date + timedelta(days=1)  # 从测试集最后一天的下一天开始
    
    for _ in range(future_steps):
        future_dates.append(current_date)
        current_date += timedelta(days=1)
    
    return future_dates

def plot_combined_predictions(args):
    """在同一张图中绘制测试集真实值、测试集预测值和未来预测值"""
    # 根据参数生成结果文件夹路径，与训练时保持一致
    result_dir = f"./results/lstm_{args.data_name}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_hd{args.hidden_dim}_nl{args.num_layers}_0"
    
    # 检查结果文件夹是否存在
    if not os.path.exists(result_dir):
        print(f"错误：结果文件夹 {result_dir} 不存在")
        return
    
    # 1. 加载所有预测结果
    try:
        y_test_true = np.load(os.path.join(result_dir, 'y_test_true.npy'))  # 形状：(样本数, pred_len)
        y_test_pred = np.load(os.path.join(result_dir, 'y_test_pred.npy'))  # 形状：(样本数, pred_len)
        future_preds = np.load(os.path.join(result_dir, 'future_predictions.npy'))  # 形状：(pred_len,)
    except FileNotFoundError as e:
        print(f"错误：在结果文件夹中未找到预测结果文件: {e}")
        return
    
    # 打印npy文件形状
    print_npy_shapes(result_dir)
    
    # 2. 处理测试集数据（拼接为完整序列）
    full_test_true = []
    full_test_pred = []
    for i in range(len(y_test_true)):
        full_test_true.extend(y_test_true[i])
        full_test_pred.extend(y_test_pred[i])
    
    # 计算测试集预测序列的总长度（非重叠）
    total_test_length = len(y_test_true) + args.pred_len - 1
    full_test_true = full_test_true[-total_test_length:]  # 取最后non-overlap的部分
    full_test_pred = full_test_pred[-total_test_length:]
    
    # 3. 获取测试集对应的日期
    test_dates = get_test_dates(args, len(y_test_true))
    
    # 确保日期长度与预测序列长度一致
    if len(test_dates) > len(full_test_true):
        test_dates = test_dates[:len(full_test_true)]
    elif len(test_dates) < len(full_test_true):
        # 如果日期不足，从测试集预测值中截取
        full_test_true = full_test_true[:len(test_dates)]
        full_test_pred = full_test_pred[:len(test_dates)]
    
    # 检查test_dates是否为空
    if len(test_dates) == 0:
        print("错误：测试集日期为空，无法生成图表")
        return
    
    # 4. 生成未来预测的日期
    last_test_date = test_dates.iloc[-1]  # 使用iloc确保按位置访问
    future_dates = generate_future_dates(last_test_date, len(future_preds))
    
    # 5. 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制测试集真实值
    plt.plot(test_dates, full_test_true, 'b-', label='测试集真实值', linewidth=2)
    
    # 绘制测试集预测值
    plt.plot(test_dates, full_test_pred, 'r-', label='测试集预测值', linewidth=2)
    
    # 绘制未来预测值
    plt.plot(future_dates, future_preds, 'g--', label='未来预测值', linewidth=2)
    
    # 添加垂直线分隔测试集和未来预测
    plt.axvline(x=last_test_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    plt.text(last_test_date, plt.ylim()[1]*0.95, '预测开始', 
             rotation=90, verticalalignment='top', fontsize=10)
    
    # 设置图表标题和标签
    plt.title(f'电力负荷预测结果 (seq_len={args.seq_len}, pred_len={args.pred_len})', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel(args.target_col, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保img目录存在
    img_dir = os.path.join(result_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # 保存图表
    save_path = os.path.join(img_dir, 'combined_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合预测图已保存至: {save_path}")

def parse_args():
    """解析命令行参数，与main_lstm.py保持一致"""
    parser = argparse.ArgumentParser(description='LSTM 模型预测结果可视化')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/', help='数据根目录')
    parser.add_argument('--data_subdir', type=str, default='ELE', help='数据子目录')
    parser.add_argument('--data_file', type=str, default='power_load.csv', help='数据文件名')
    parser.add_argument('--data_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--features', type=str, required=True, help='特征类型: S(单变量)')
    parser.add_argument('--target_col', type=str, required=True, help='目标列名')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, required=True, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, required=True, help='预测序列长度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout率')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 调用组合可视化函数
    plot_combined_predictions(args)