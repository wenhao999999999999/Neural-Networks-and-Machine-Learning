import torch
import torch.nn as nn
import pandas as pd

class LSTMForecaster(nn.Module):
    def __init__(self, args):
        super(LSTMForecaster, self).__init__()
        self.args = args
        
        # 动态计算输入维度（单变量=1，多变量=特征列数）
        if args.features == 'S':
            self.input_dim = 1
        else:
            # 多变量时，读取数据集获取特征数量（需提前加载数据）
            # 更严谨的方式：从 data_loader 中传递 input_dim，这里简化处理
            dummy_df = pd.read_csv(f"{args.data_dir}/{args.data_name}/{args.data_name}.csv")
            self.input_dim = len(dummy_df.drop(columns=['date']).columns)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout if args.num_layers > 1 else 0
        )
        
        # 输出层（预测 pred_len 个时间步）
        self.fc = nn.Linear(args.hidden_dim, args.pred_len)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)  # 输出: [batch, seq_len, hidden_dim]
        return self.fc(out[:, -1, :])  # 取最后一个时间步预测