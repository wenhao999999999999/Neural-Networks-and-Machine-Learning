import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

def per_series_metrics(y_true, y_pred, series_names, eps=1e-6):
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    res = {}
    for i, name in enumerate(series_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        res[name] = {
            'MAE': mae(yt, yp),
            'RMSE': rmse(yt, yp),
            'MAPE': mape(yt, yp, eps),
        }
    res['OVERALL'] = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred, eps),
    }
    return res
