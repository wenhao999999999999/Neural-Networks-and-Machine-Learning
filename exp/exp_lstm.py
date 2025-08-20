# exp/exp_lstm.py

import os
import torch
import numpy as np
import time
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.data_loader import get_dataloader, TimeSeriesDataset
from models.lstm_model import LSTMForecaster
from utils.tools import calculate_metrics, plot_predictions

class Exp_LSTM:
    """LSTM 实验类，仿照 Informer 的结构"""
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        """获取设备（GPU或CPU）"""
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: {device}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        """构建模型"""
        model = LSTMForecaster(self.args)
        return model
    
    def _load_and_preprocess_data(self, flag):
        """加载和预处理数据"""
        from data.data_loader import load_and_preprocess_data, split_dataset
        
        # 加载原始数据
        raw_data, self.raw_df = load_and_preprocess_data(self.args)
        
        # 划分数据集
        if flag == 'train':
            # 对于训练集，先划分再归一化
            train_data = split_dataset(raw_data, self.args, 'train')
            
            # 只在训练集上拟合归一化器
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            norm_train_data = self.scaler.fit_transform(train_data)
            
            return norm_train_data, train_data
        
        elif flag == 'val':
            # 对于验证集，使用训练集的归一化器
            val_data = split_dataset(raw_data, self.args, 'val')
            norm_val_data = self.scaler.transform(val_data) if hasattr(self, 'scaler') else val_data
            
            return norm_val_data, val_data
        
        elif flag == 'test':
            # 对于测试集，使用训练集的归一化器
            test_data = split_dataset(raw_data, self.args, 'test')
            norm_test_data = self.scaler.transform(test_data) if hasattr(self, 'scaler') else test_data
            
            return norm_test_data, test_data
        
        else:  # 'full'
            # 对于完整数据集，使用训练集的归一化器
            full_data = split_dataset(raw_data, self.args, 'full')
            norm_full_data = self.scaler.transform(full_data) if hasattr(self, 'scaler') else full_data
            
            return norm_full_data, full_data

    def _get_data(self, flag):
        """获取数据加载器"""
        from data.data_loader import get_dataloader
        
        # 获取处理后的数据
        norm_data, raw_data = self._load_and_preprocess_data(flag)
        
        # 创建数据加载器
        data_loader = get_dataloader(norm_data, self.args, flag=flag)
    
        # 创建一个简单的数据集对象，用于存储原始数据和归一化器
        class SimpleDataset:
            def __init__(self, data, scaler):
                self.data = data
                self.scaler = scaler
                self.raw_data = raw_data  # 存储原始数据（未归一化）
        
        dataset = SimpleDataset(norm_data, self.scaler)
        
        return data_loader, dataset

    
    def _select_optimizer(self):
        """选择优化器"""
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    
    def _select_criterion(self):
        """选择损失函数"""
        return torch.nn.MSELoss()
    
    def _create_experiment_dir(self, setting):
        """创建实验目录"""
        exp_dir = os.path.join('results', setting)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'img'), exist_ok=True)
        return exp_dir
    
    def _get_dataset_predictions(self, data_loader):
        """获取指定数据集的预测结果"""
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device).float()
                pred = self.model(x).cpu().numpy()
                y_true.append(y.numpy())
                y_pred.append(pred)
                
        # 修改这里：确保正确拼接所有批次的预测结果
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        return y_true, y_pred
    
    def train(self, setting):
        # 创建实验目录
        exp_dir = self._create_experiment_dir(setting)
        print(f"实验结果将保存至: {exp_dir}")
        
        # 加载数据 - 先加载训练数据以拟合归一化器
        train_loader, train_dataset = self._get_data(flag='train')
        
        # 然后加载验证和测试数据（使用训练集的归一化器）
        val_loader, _ = self._get_data(flag='val')
        test_loader, test_dataset = self._get_data(flag='test')
        
        # 保存训练集的 scaler
        self.train_scaler = train_dataset.scaler

        print(f'训练集样本数: {len(train_loader.dataset)}')
        print(f'验证集样本数: {len(val_loader.dataset)}')
        print(f'测试集样本数: {len(test_loader.dataset)}')
        
        # 初始化优化器和损失函数
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        
        best_val_loss = np.inf
        total_start_time = time.time()
        
        # 早停机制
        patience = self.args.patience
        no_improve_count = 0
        train_losses, val_losses = [], []
        
        # 训练循环
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device).float(), y.to(self.device).float()
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # 时间计算
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - total_start_time
            remaining_time = (self.args.epochs - epoch - 1) * epoch_time
            
            # 保存最优模型
            model_save_path = os.path.join(exp_dir, f'{setting}_best.pth')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                no_improve_count = 0
                print(f'Epoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                      f'Time: {timedelta(seconds=int(epoch_time))} | ETA: {timedelta(seconds=int(remaining_time))} | 已保存最优模型')
            else:
                no_improve_count += 1
                print(f'Epoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                      f'Time: {timedelta(seconds=int(epoch_time))} | ETA: {timedelta(seconds=int(remaining_time))}')

            # 早停判断
            if no_improve_count >= patience:
                print(f"早停触发：在第 {epoch+1} 轮停止，验证集损失连续 {patience} 轮未改善")
                break
        
        # 总训练时间
        total_time = time.time() - total_start_time
        print(f'训练完成。总耗时: {timedelta(seconds=int(total_time))}')
        
        # 加载最优模型
        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device, weights_only=True))
        
        # 保存训练信息
        self._save_training_info(exp_dir, train_losses, val_losses, total_time, epoch+1-no_improve_count)
        
        return exp_dir
    
    def _save_training_info(self, exp_dir, train_losses, val_losses, total_time, best_epoch):
        """保存训练信息"""
        # 保存损失曲线
        loss_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
        loss_df.to_csv(os.path.join(exp_dir, 'losses.csv'), index=False)
        
        # 保存参数配置
        with open(os.path.join(exp_dir, 'args.txt'), 'w') as f:
            for arg in vars(self.args):
                f.write(f'{arg}: {getattr(self.args, arg)}\n')
    
    def test(self, setting, test_loader=None, test_dataset=None):
        """测试模型"""
        # 如果没有提供数据，则加载数据
        if test_loader is None or test_dataset is None:
            test_loader, test_dataset = self._get_data(flag='test')
        
        # 获取测试集预测结果
        y_test_true, y_test_pred = self._get_dataset_predictions(test_loader)
        
        # 反归一化 - 使用训练集的归一化器
        scaler = self.train_scaler
        
        # 确保我们有原始数据用于反归一化
        if hasattr(test_dataset, 'raw_data'):
            # 计算原始数据的形状
            raw_data_shape = test_dataset.raw_data.shape[1] if len(test_dataset.raw_data.shape) > 1 else 1
            
            # 反归一化
            if len(y_test_true.shape) == 1 or y_test_true.shape[1] == 1:
                y_test_true_denorm = scaler.inverse_transform(y_test_true.reshape(-1, 1)).flatten()
                y_test_pred_denorm = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            else:
                # 对于多变量情况，需要确保形状匹配
                y_test_true_denorm = scaler.inverse_transform(
                    y_test_true.reshape(-1, raw_data_shape)
                ).reshape(y_test_true.shape)
                y_test_pred_denorm = scaler.inverse_transform(
                    y_test_pred.reshape(-1, raw_data_shape)
                ).reshape(y_test_pred.shape)
        else:
            # 备用方案：如果无法获取原始数据形状
            if len(y_test_true.shape) == 1 or y_test_true.shape[1] == 1:
                y_test_true_denorm = scaler.inverse_transform(y_test_true.reshape(-1, 1)).flatten()
                y_test_pred_denorm = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            else:
                y_test_true_denorm = scaler.inverse_transform(y_test_true.reshape(-1, y_test_true.shape[-1]))
                y_test_pred_denorm = scaler.inverse_transform(y_test_pred.reshape(-1, y_test_pred.shape[-1]))
                y_test_true_denorm = y_test_true_denorm.reshape(y_test_true.shape)
                y_test_pred_denorm = y_test_pred_denorm.reshape(y_test_pred.shape)        
        # 计算评估指标
        test_mse, test_mae, test_rmse = calculate_metrics(
            y_test_true, y_test_pred, scaler, self.args)
        
        metrics = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        
        # 保存结果
        exp_dir = os.path.join('results', setting)
        np.save(os.path.join(exp_dir, 'y_test_true.npy'), y_test_true_denorm)
        np.save(os.path.join(exp_dir, 'y_test_pred.npy'), y_test_pred_denorm)
        
        # 保存评估指标
        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join(exp_dir, 'test_metrics.csv'), index=False)
        
        print(f"测试完成。MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        return metrics
    
    def predict(self, setting, future_steps=None):
        """预测未来值"""
        
        # 加载完整数据集
        full_norm_data, full_dataset = self._load_and_preprocess_data(flag='full')
        
        # 获取最后一个序列
        last_sequence = full_norm_data[-self.args.seq_len:]
        
        # 预测未来值
        if future_steps is None:
            future_steps = self.args.pred_len
            
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            input_seq = torch.tensor(
                last_sequence.reshape(1, self.args.seq_len, -1), 
                dtype=torch.float32
            ).to(self.device)
            
            # 直接预测完整序列
            future_preds = self.model(input_seq).cpu().numpy()[0]
        
        # 反归一化
        scaler = self.train_scaler
        future_preds = np.array(future_preds)
        
        # 获取原始数据形状
        raw_data_shape = full_dataset.shape[1] if len(full_dataset.shape) > 1 else 1
        
        # 反归一化
        if len(future_preds.shape) == 1 or future_preds.shape[1] == 1:
            future_preds_denorm = scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten()
        else:
            future_preds_denorm = scaler.inverse_transform(
                future_preds.reshape(-1, raw_data_shape)
            )
            future_preds_denorm = future_preds_denorm.reshape(future_preds.shape)
        
        # 保存预测结果
        exp_dir = os.path.join('results', setting)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        np.save(os.path.join(exp_dir, 'future_predictions.npy'), future_preds_denorm)
        
        print(f"已完成 {future_steps} 步未来预测")
        
        return future_preds_denorm