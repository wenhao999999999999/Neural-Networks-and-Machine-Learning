import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils.tools import aggregate_hourly_to_daily  # 保留小时转日级的函数

class TimeSeriesDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.features = args.features
        self.target_col = args.target_col
        
        # 1. 加载并聚合数据（小时级→日级）
        data_path = f"{args.data_dir}/{args.data_subdir}/{args.data_file}"
        self.df = aggregate_hourly_to_daily(data_path)  # 已确认返回正确的日级数据
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 检查聚合后的数据量
        total_days = len(self.df)
        print(f"聚合后总天数: {total_days}")  # 打印总天数，方便调试
        if total_days < self.seq_len + self.pred_len:
            raise ValueError(
                f"数据量不足！需要至少 {self.seq_len + self.pred_len} 天数据，"
                f"但仅聚合到 {total_days} 天"
            )
        
        # 2. 特征选择（单变量）
        if self.features == 'S':
            # 确保目标列存在
            if self.target_col not in self.df.columns:
                raise ValueError(f"目标列 {self.target_col} 不在数据中，请检查列名")
            self.data = self.df[[self.target_col]].values
        else:
            self.data = self.df.drop(columns=['date']).values
        
        # 3. 归一化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        
        # 4. 划分数据集（动态调整比例，确保每个子集能生成样本）
        self.data = self._split_dataset()
        
        # 5. 构建时序样本
        self.x, self.y = self._build_sequences()
        
        # 检查样本数
        if len(self.x) == 0:
            raise ValueError(
                f"{self.flag}集无法生成样本！"
                f"需要至少 {self.seq_len + self.pred_len} 条数据，实际有 {len(self.data)} 条"
            )
    
    def _split_dataset(self):
        """动态划分数据集，确保每个子集都能生成样本"""
        total_len = len(self.data)
        min_required = self.seq_len + self.pred_len  # 每个子集需要的最小长度
        
        # 计算测试集和验证集的长度（至少满足最小需求，最多不超过总长度的30%）
        test_len = min(
            max(min_required, int(total_len * 0.1)),  # 至少min_required，最多10%
            total_len - 2 * min_required  # 预留足够数据给训练集和验证集
        )
        val_len = min(
            max(min_required, int(total_len * 0.2)),  # 至少min_required，最多20%
            total_len - test_len - min_required  # 预留足够数据给训练集
        )
        train_len = total_len - val_len - test_len  # 剩余给训练集
        
        # 打印划分结果，方便调试
        print(f"数据划分: 训练集{train_len}条, 验证集{val_len}条, 测试集{test_len}条")
        
        # 按flag返回对应子集
        if self.flag == 'train':
            return self.data[:train_len]
        elif self.flag == 'val':
            return self.data[train_len : train_len + val_len]
        else:  # test
            return self.data[train_len + val_len :]
    
    def _build_sequences(self):
        """生成输入序列x和预测序列y"""
        x_list, y_list = [], []
        # 计算可生成的样本数量
        max_index = len(self.data) - self.seq_len - self.pred_len + 1
        
        for i in range(max_index):
            # 输入序列：[i, i+seq_len)
            x = self.data[i : i + self.seq_len]
            # 预测序列：[i+seq_len, i+seq_len+pred_len)，取第一列（目标列）
            y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len, 0]
            x_list.append(x)
            y_list.append(y)
        
        print(f"{self.flag}集生成样本数: {len(x_list)}")  # 打印样本数
        return np.array(x_list), np.array(y_list)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(args, flag='train', batch_size=None):
    """创建数据加载器"""
    dataset = TimeSeriesDataset(args, flag)
    batch_size = batch_size or args.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=2,
        pin_memory=True
    )
