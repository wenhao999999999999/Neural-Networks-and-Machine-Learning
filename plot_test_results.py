import os
import numpy as np
import pandas as pd
import argparse
from utils.tools import plot_predictions, aggregate_hourly_to_daily
from exp.exp_lstm import Exp_LSTM

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

def get_test_dates(args):
    """获取测试集样本对应的日期范围"""
    # 加载原始数据（日级聚合后的数据）
    data_path = f"{args.data_dir}/{args.data_subdir}/{args.data_file}"
    raw_df = aggregate_hourly_to_daily(data_path)
    raw_dates = raw_df['date']  # 完整日期序列（datetime格式）
    
    # 划分数据集（复用split_dataset逻辑计算测试集在原始数据中的位置）
    total_len = len(raw_df)
    min_required = args.seq_len + args.pred_len
    # 计算测试集长度（与data_loader.split_dataset保持一致）
    test_len = min(
        max(min_required, int(total_len * 0.15)),
        total_len - 2 * min_required
    )
    val_len = min(
        max(min_required, int(total_len * 0.15)),
        total_len - test_len - min_required
    )
    train_len = total_len - val_len - test_len
    
    # 测试集在原始数据中的起始索引（训练集结束 + 验证集结束）
    test_start_idx = train_len + val_len
    # 测试集样本覆盖的日期范围：从第一个样本的输入序列结束，到最后一个样本的预测序列结束
    first_pred_start = test_start_idx + args.seq_len  # 第一个样本的预测起始位置
    last_pred_end = test_start_idx + test_len  # 最后一个样本的预测结束位置（因test_len = 测试集数据长度）
    test_dates = raw_dates[first_pred_start : last_pred_end]
    
    return test_dates

def plot_test_predictions(args):
    """可视化测试集所有样本的预测值和真实值"""
    # 根据参数生成结果文件夹路径，与训练时保持一致
    result_dir = f"./results/lstm_ELE_S_sl60_pl7_hd64_nl2_0"
    
    # 检查结果文件夹是否存在
    if not os.path.exists(result_dir):
        print(f"错误：结果文件夹 {result_dir} 不存在")
        return
    
    # 1. 加载预测结果
    try:
        y_true = np.load(os.path.join(result_dir, 'y_test_true.npy'))  # 形状：(样本数, pred_len)
        y_pred = np.load(os.path.join(result_dir, 'y_test_pred.npy'))  # 形状：(样本数, pred_len)
    except FileNotFoundError as e:
        print(f"错误：在结果文件夹中未找到预测结果文件: {e}")
        return
    
    # 打印npy文件形状
    print_npy_shapes(result_dir)
    
    # 2. 拼接为完整序列（处理滑动窗口重叠）
    # 例如：样本1预测[1-7天]，样本2预测[2-8天]，拼接后保留所有时间步
    full_true = []
    full_pred = []
    for i in range(len(y_true)):
        # 每个样本的预测序列从第i个时间步开始（相对于测试集预测起始点）
        full_true.extend(y_true[i])
        full_pred.extend(y_pred[i])
    
    # 3. 去重（滑动窗口重叠部分会重复，保留最后一个值）
    # 计算测试集预测序列的总长度（非重叠）
    total_pred_length = len(y_true) + args.pred_len - 1
    full_true = full_true[-total_pred_length:]  # 取最后non-overlap的部分
    full_pred = full_pred[-total_pred_length:]
    
    # 4. 获取对应的日期
    test_dates = get_test_dates(args)
    # 确保日期长度与预测序列长度一致
    test_dates = test_dates[:len(full_true)]
    
    # 5. 调用绘图工具
    plot_predictions(
        y_true=full_true,
        y_pred=full_pred,
        dates=test_dates,
        title=f"测试集预测结果 (seq_len={args.seq_len}, pred_len={args.pred_len})",
        label=args.target_col,  # 使用目标列名作为标签
        save_path=os.path.join(result_dir,'img', 'test_predictions.png')
    )
    print(f"测试集可视化结果已保存至: {os.path.join(result_dir, 'img',  'test_predictions.png')}")

def parse_args():
    """解析命令行参数，与main_lstm.py保持一致"""
    parser = argparse.ArgumentParser(description='LSTM 模型测试集预测结果可视化')
    
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
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--do_predict', type=bool, default=False, help='是否进行预测')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 调用可视化函数
    plot_test_predictions(args)
