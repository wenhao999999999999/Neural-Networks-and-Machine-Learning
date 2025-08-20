# main_lstm.py

import argparse
import os
import torch
import random
import numpy as np

from exp.exp_lstm import Exp_LSTM

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM Time Series Forecasting')
    
    # 数据配置
    parser.add_argument('--data_dir', type=str, default='./data/', help='数据根目录')
    parser.add_argument('--data_subdir', type=str, default='ELE', help='数据集子目录名')
    parser.add_argument('--data_file', type=str, default='power_load.csv', help='CSV文件名')
    parser.add_argument('--data_name', type=str, default='ELE', help='数据集名称（用于保存路径）')
    parser.add_argument('--features', type=str, default='S', choices=['S', 'M'], help='S:单变量, M:多变量')
    parser.add_argument('--target_col', type=str, default='daily_power_load', help='单变量目标列')
    parser.add_argument('--label_index', type=int, default=0, help='标签列的索引')
    
    # 序列与模型配置
    parser.add_argument('--seq_len', type=int, default=123, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=31, help='预测序列长度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM隐藏维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停轮数')
    
    # 设备配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    
    # 预测配置
    parser.add_argument('--do_predict', action='store_true', help='是否进行未来预测', default=False)
    
    # 实验次数
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 固定随机种子以确保结果可复现
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args = parse_args()
    
    # 设备配置
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.deterministic = True
    
    print('实验参数:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    # 多次实验循环
    for ii in range(args.itr):
        # 生成实验设置标识
        setting = 'lstm_{}_{}_sl{}_pl{}_hd{}_nl{}_{}'.format(
            args.data_name, args.features, args.seq_len, args.pred_len,
            args.hidden_dim, args.num_layers, ii
        )
        
        # 创建实验实例
        exp = Exp_LSTM(args)
        
        # 训练模型
        print(f'>>>>>>>开始训练: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        
        # 测试模型
        print(f'>>>>>>>开始测试: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
        
        # 如果需要，进行预测
        if args.do_predict:
            print(f'>>>>>>>开始预测: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting)
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()