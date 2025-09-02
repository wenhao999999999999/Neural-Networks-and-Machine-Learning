LSTM 多目标时间序列预测（湛江项目）

简介
- 基于 PyTorch 的多目标时间序列预测范例，面向“宽表”数据：一列日期 `date` + 多个地区/品类列（本工程默认 6 个区县）。
- 通过滑动窗口构造样本，使用 LSTM 回归同时预测多个目标列；支持时间特征、缺失值按日历修复、归一化、评估与可视化。
- 主入口见 `main_lstm.py:1`，配置见 `config/config.yaml:1`。

快速开始
- 环境：Python 3.8+（推荐 3.9/3.10），Windows/Linux/macOS 均可。GPU 可选。
- 安装依赖（CPU 版本）:
  - `pip install -r requirements.txt`
  - 若需 CUDA，请参考 PyTorch 官网选择与你 CUDA 版本匹配的安装命令。
- 一键运行（训练+评估+预测+绘图）：
  - `python main_lstm.py --config config/config.yaml --run_name with_weekday_cycle --use_fill_calendar --use_scaler --do_train --do_eval --do_predict --do_plot`

目录结构
- `main_lstm.py:1`：主脚本，串联数据处理、建模、训练、评估、预测、绘图。
- `config/config.yaml:1`：统一配置文件（数据路径、滑窗、模型、训练、输出等）。
- `preprocess/preprocessing.py:1`：时间特征、日历缺失修复、列级 MinMax 缩放器。
- `preprocess/slicing.py:1`：多目标（宽表）滑窗构造。
- `dataset/timeseries_dataset.py:1`：PyTorch Dataset 封装。
- `models/lstm.py:1`：LSTM 预测器（最后一步隐藏状态接全连接）。
- `training/trainer.py:1`：训练循环、早停、学习率调度与最优权重保存。
- `tools/evaluator.py:1`：测试集推断、反归一化、指标计算与落盘。
- `tools/predictor.py:1`：未来递归预测与落盘。
- `tools/plot_combined.py:1`、`tools/plot_utils.py:1`：绘图工具（单区县/合计曲线）。
- `tools/metrics.py:1`：MAE/MSE/RMSE/MAPE 等评估指标。
- `data/`：示例数据目录（请放置你的 CSV）。
- `outputs/`：运行产物（日志、预测、图片、模型权重）。

数据与配置
- 输入数据（宽表）：CSV 至少包含：
  - 一列日期：默认名 `date`（可在配置中改为其他列名）。
  - 多个目标列：默认 6 个区县列，见 `config.yaml` 的 `data.series_cols`。
- 关键配置项（`config/config.yaml:1`）：
  - `data.csv_path`：输入 CSV 路径（示例：`data/sales_cleaned_stl.csv`）。
  - `data.date_col`：日期列名。
  - `data.series_cols`：需要同时作为“输入与预测目标”的多列（多目标）。
  - `data.add_time_features / weekday_cyclical / month_cyclical`：是否添加时间特征以及采用周期编码。
  - `window.seq_len / pred_len / stride`：滑窗长度、预测步数与滑动步长。
  - `split.train_ratio / val_ratio / test_ratio`：数据集划分比例。
  - `model.hidden_dim / num_layers / dropout`：LSTM 结构超参；`input_size` 自动推断。
  - `train.device`：`cuda` 或 `cpu`；脚本会在不可用时回退至 `cpu`。
  - `train.batch_size / epochs / lr / patience / grad_clip`：训练相关超参。
  - `eval.metrics`：评估指标列表。
  - `predict.horizon`：未来滚动预测步数（天）。
  - `paths.outputs / figures`：输出目录与绘图目录。
  - `run_name`：本次运行标识，会拼接到输出路径中。

运行参数（主脚本）
- `--config`：配置文件路径。
- `--run_name`：覆盖配置中的运行名（可用于区分多次实验）。
- `--use_fill_calendar`：按日历补齐与修复缺失（休息日置 0，工作日线性插值，再做前后向填充与非负裁剪）。见 `preprocess/preprocessing.py:1`。
- `--use_scaler`：使用列级 MinMax 缩放器（训练、评估、预测时保持一致）。见 `preprocess/preprocessing.py:1`。
- `--do_train / --do_eval / --do_predict / --do_plot`：分别执行训练、测试集评估、未来预测与绘图。

输出说明
- 位置：`outputs/runs/<run_name>/`
  - `checkpoints/best.pt`：验证集最优模型权重。
  - `logs/config_snapshot.json`：运行时配置快照。
  - `logs/eval_metrics.json`：评估指标（MAE/MSE/RMSE/MAPE）。
  - `logs/scaler.json`：缩放器（若启用）。
  - `predictions/test_predictions.csv`：测试集逐日逐列真值与预测，已尽可能反归一化。
  - `predictions/future_forecast.csv`：未来预测逐日逐列，已尽可能反归一化。
- 图片：
  - 单区县：`outputs/figures/<run_name>/*.png`，由 `tools/plot_combined.py:1` 的 `plot_per_series` 产出（在主脚本中调用）。
  - 合计曲线：可调用 `tools/plot_combined.py:1` 的 `plot_combined` 产出总和图。

工作流（建议）
1) 准备数据：将你的宽表 CSV 放到 `data/` 并在 `config/config.yaml` 中设置路径与列名。
2) 启用日历修复与时间特征：`--use_fill_calendar --use_scaler` 一般更稳健。
3) 调整滑窗与模型超参：`window.*` 与 `model.*`。
4) 运行训练与评估：`--do_train --do_eval`，检查 `eval_metrics.json`。
5) 进行未来预测与绘图：`--do_predict --do_plot`，查看 `predictions/` 和 `outputs/figures/`。

常见问题
- 未找到权重：直接评估/预测时报错“未找到已训练权重”：请先加 `--do_train` 或确保 `checkpoints/best.pt` 已存在。见 `main_lstm.py:1`。
- CUDA 不可用：当 `train.device=cuda` 但本机无 GPU 时，脚本会自动回退到 CPU。见 `training/trainer.py:1` 与相关用法。
- chinese_calendar 可选：若未安装，会自动将双休日视为休息日；安装后可依据中国节假日法定日历更精确。见 `preprocess/preprocessing.py:1` 与 `tools/predictor.py:1`。
- 可视化列名不匹配：确保 `test_predictions.csv` 中存在 `y_true_/y_pred_`（或 `*_inv_*`）列，`future_forecast.csv` 中存在 `y_pred_`（或 `y_pred_inv_`）列。见 `tools/plot_combined.py:1`。
- 超参搜索脚本：`tools/tune_lstm.py:1` 仅作示例，依赖 `optuna` 且与当前滑窗接口不完全一致，如需使用请自行调整并安装 `optuna`。

致谢与许可
- 本工程为内部项目示例模板，若需对外发布请补充相应的许可与声明。
