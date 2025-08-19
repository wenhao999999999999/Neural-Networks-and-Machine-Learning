### S
python -u main_lstm.py --features S --data_name ELE --target_col daily_power_load --seq_len 123 --pred_len 31 --hidden_dim 64 --num_layers 2 --batch_size 32 --epochs 100 --lr 1e-3

python -u main_lstm.py --features S --data_name ELE --target_col daily_power_load --seq_len 62 --pred_len 7 --hidden_dim 64 --num_layers 2 --batch_size 32 --epochs 100 --lr 1e-3