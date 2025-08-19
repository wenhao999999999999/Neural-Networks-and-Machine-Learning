import argparse
import torch
import numpy as np
import time
from datetime import timedelta
from data.data_loader import get_dataloader
from models.lstm_model import LSTMForecaster
from utils.tools import calculate_metrics, plot_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM Time Series Forecasting')
    # 数据配置
    parser.add_argument('--data_dir', type=str, default='data', help='数据根目录')
    parser.add_argument('--data_subdir', type=str, default='ELE', help='数据集子目录名')
    parser.add_argument('--data_file', type=str, default='power_load.csv', help='CSV文件名')
    parser.add_argument('--data_name', type=str, default='ELE', help='数据集名称（用于保存路径）')
    parser.add_argument('--features', type=str, choices=['S', 'M'], required=True, help='S:单变量, M:多变量')
    parser.add_argument('--target_col', type=str, default='daily_power_load', help='单变量目标列')
    
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
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    # 早停轮数
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    
    return parser.parse_args()

def train(args):
    # 设备配置与显示
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Use GPU: {device}')
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # 加载数据并显示样本数
    train_loader = get_dataloader(args, flag='train')
    val_loader = get_dataloader(args, flag='val', batch_size=args.batch_size*2)
    test_loader = get_dataloader(args, flag='test', batch_size=args.batch_size*2)
    
    print(f'train {len(train_loader.dataset)}')
    print(f'val {len(val_loader.dataset)}')
    print(f'test {len(test_loader.dataset)}')

    # 初始化模型
    model = LSTMForecaster(args).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练信息显示
    model_name = f'lstm_{args.data_name}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_hd{args.hidden_dim}'
    print(f'>>>>>>>start training : {model_name}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    best_val_loss = np.inf
    total_start_time = time.time()  # 总训练时间计时
    
    # 添加早停机制
    patience = args.patience  # 允许验证损失未改善的最大轮次
    no_improve_count = 0  # 未改善的轮次计数

    for epoch in range(args.epochs):
        epoch_start_time = time.time()  # 每轮开始时间
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        # 计算耗时与剩余时间
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time
        remaining_time = (args.epochs - epoch - 1) * epoch_time
        
        # 保存最优模型
        save_path = f'results/{model_name}_best.pth'
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve_count = 0  # 重置计数器
            print(f'Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Time: {timedelta(seconds=int(epoch_time))} | ETA: {timedelta(seconds=int(remaining_time))} | Saved to {save_path}')
        else:
            no_improve_count += 1
            print(f'Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Time: {timedelta(seconds=int(epoch_time))} | ETA: {timedelta(seconds=int(remaining_time))}')

        # 检查是否需要早停
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} consecutive epochs.")
            break
    
    # 总训练时间
    total_time = time.time() - total_start_time
    print(f'Training completed. Total time: {timedelta(seconds=int(total_time))}')
    
    # 测试阶段
    model.load_state_dict(torch.load(save_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad(): # 关闭自动求导机制
        for x, y in test_loader:
            x = x.to(device).float()
            pred = model(x).cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(pred)
    
    # 计算指标
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    mse, mae, rmse = calculate_metrics(y_true, y_pred, train_loader.dataset.scaler, args)
    print(f'Test Metrics | MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    
    # 可视化
    plot_predictions(y_true, y_pred, train_loader.dataset.scaler, args, 
                     save_path=f'img/{model_name}_pred.png')

if __name__ == '__main__':
    args = parse_args()
    # 打印所有参数
    print('Args in experiment:')
    print(args)
    train(args)
