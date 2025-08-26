# 湛江市各区县卷烟销量 31 天预测（PyTorch）


- 多变量 → 多步（31 天）预测；
- 节假日缺失填 0，工作日缺失插值；
- 自带特征：是否节假日、周几（月度）周期编码、是否月末；
- 一键训练 + 评估 + 预测，输出 CSV 与指标 JSON。


## 快速开始
```bash
pip install -r requirements.txt
python src/train_eval.py --config config.yaml