# 训练
python src/train_eval.py --config config.yaml  

python src/train_eval.py --config config.yaml --run_name best_retrain


python src/plot_results.py --config config.yaml --pred_csv ./outputs/predictions/preds_20250826_143144.csv

python src/forecast_next.py --config config.yaml --ckpt outputs/checkpoints/best_model.pt
# 或自定义天数（需 ≤ pred_len）
python src/forecast_next.py --config config.yaml --ckpt outputs/checkpoints/best_model.pt --days 31

python src/plot_combined.py --config config.yaml --hist_pred_csv ./outputs/predictions/preds_20250826_143144.csv --future_pred_csv ./outputs/predictions/future_31d_20250826_161757.csv

# 结果将保存到 outputs/compare_predlen/<时间戳>/ 下
python src/compare_predlen_suite.py --config config.yaml --seq_len 60 --hidden_dim 128 --num_layers 2


# 固定 pred_len=1，对比 30/60/90/120/180
python src/tune_lstm.py --config config.yaml --metric RMSE --n_trials 30
