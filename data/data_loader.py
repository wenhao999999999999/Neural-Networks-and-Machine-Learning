import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.tools import aggregate_hourly_to_daily

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, label_index=0):
        """
        初始化数据集
        Args:
            data: 已经预处理和归一化后的数据
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            label_index: 标签列的索引
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_index = label_index
        
        # 构建时序样本
        self.x, self.y = self._build_sequences()
        
        # 检查样本数
        if len(self.x) == 0:
            raise ValueError(
                f"无法生成样本！需要至少 {self.seq_len + self.pred_len} 条数据，实际有 {len(self.data)} 条"
            )
    
    def _build_sequences(self):
        """生成输入序列x和预测序列y"""
        x_list, y_list = [], []
        # 计算可生成的样本数量
        max_index = len(self.data) - self.seq_len - self.pred_len + 1
        
        for i in range(max_index):
            # 输入序列：[i, i+seq_len)
            x = self.data[i : i + self.seq_len]
            # 预测序列：[i+seq_len, i+seq_len+pred_len)，取指定列（目标列）
            y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len, self.label_index]
            x_list.append(x)
            y_list.append(y)
        
        return np.array(x_list), np.array(y_list)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_and_preprocess_data(args, flag='train'):
    """加载和预处理数据"""
    # 1. 加载并聚合数据（小时级→日级）
    data_path = f"{args.data_dir}/{args.data_subdir}/{args.data_file}"
    df = aggregate_hourly_to_daily(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. 检查聚合后的数据量
    total_days = len(df)
    if total_days < args.seq_len + args.pred_len:
        raise ValueError(
            f"数据量不足！需要至少 {args.seq_len + args.pred_len} 天数据，"
            f"但仅聚合到 {total_days} 天"
        )
    
    # 3. 特征选择
    if args.features == 'S':
        if args.target_col not in df.columns:
            raise ValueError(f"目标列 {args.target_col} 不在数据中，请检查列名")
        data = df[[args.target_col]].values
    else:
        data = df.drop(columns=['date']).values
    
    return data, df

def split_dataset(data, args, flag='train'):
    """划分数据集"""
    total_len = len(data)
    min_required = args.seq_len + args.pred_len
    
    # 计算测试集和验证集的长度
    test_len = min(
        max(min_required, int(total_len * 0.15)),
        total_len - 2 * min_required
    )
    val_len = min(
        max(min_required, int(total_len * 0.15)),
        total_len - test_len - min_required
    )
    train_len = total_len - val_len - test_len
    
    # 按flag返回对应子集
    if flag == 'train':
        return data[:train_len]
    elif flag == 'val':
        return data[train_len : train_len + val_len]
    elif flag == 'test':
        return data[train_len + val_len :]
    else:  # 'full' 用于预测
        return data

def get_dataloader(data, args, flag='train', batch_size=None):
    """创建数据加载器"""
    dataset = TimeSeriesDataset(data, args.seq_len, args.pred_len, args.label_index)
    batch_size = batch_size or args.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=2,
        pin_memory=True
    )