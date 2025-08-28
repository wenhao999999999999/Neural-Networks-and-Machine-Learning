# -*- coding: utf-8 -*-
import numpy as np

def mae(y, yhat):  return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))

def mse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) ** 2))

def rmse(y, yhat): return float(np.sqrt(mse(y, yhat)))

def mape(y, yhat, eps: float = 1e-8):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / (np.abs(y) + eps))) * 100.0)

# 更稳健：忽略极小真值
def mape_safe(y, yhat, eps: float = 1e-8, min_abs_y: float = 1e-3):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.abs(y) >= min_abs_y
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / (np.abs(y[mask]) + eps))) * 100.0)

def smape(y, yhat, eps: float = 1e-8):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat)) + eps
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100.0)
