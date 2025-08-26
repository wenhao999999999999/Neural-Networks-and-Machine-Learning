python src/train_eval.py --config config.yaml    

python src/plot_results.py --config config.yaml --pred_csv ./outputs/predictions/preds_20250826_143144.csv

python src/forecast_next.py --config config.yaml --ckpt outputs/checkpoints/best_model.pt
# 或自定义天数（需 ≤ pred_len）
python src/forecast_next.py --config config.yaml --ckpt outputs/checkpoints/best_model.pt --days 31

python src/plot_combined.py --config config.yaml --hist_pred_csv ./outputs/predictions/preds_20250826_143144.csv --future_pred_csv ./outputs/predictions/future_31d_20250826_161757.csv


