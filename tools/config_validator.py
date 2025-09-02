# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path


def _assert(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def _as_positive_int(x, name: str) -> int:
    _assert(isinstance(x, int) and x > 0, f"配置项 {name} 必须为正整数")
    return x


def validate_and_fill(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    对 config.yaml 加工：
    - 校验关键字段存在与基本取值范围
    - 填充合理默认项（若缺失）
    - 标准化部分路径
    返回规范化后的 cfg（原地修改并返回）
    """
    required_top = [
        "data", "window", "split", "model", "train", "eval", "predict", "paths"
    ]
    for k in required_top:
        _assert(k in cfg, f"配置缺少顶级字段: {k}")

    # data
    data = cfg["data"]
    _assert("csv_path" in data, "data.csv_path 未设置")
    _assert("date_col" in data, "data.date_col 未设置")
    _assert("series_cols" in data and isinstance(data["series_cols"], (list, tuple)) and len(data["series_cols"]) > 0,
            "data.series_cols 必须为非空列表（多目标宽表列）")
    # 不强制文件存在以便容器镜像构建，但给出宽松提示（由主流程在运行时检查）
    data.setdefault("add_time_features", True)
    data.setdefault("weekday_cyclical", False)
    data.setdefault("month_cyclical", True)
    data.setdefault("freq", "D")

    # window
    window = cfg["window"]
    window["seq_len"] = _as_positive_int(window.get("seq_len", 15), "window.seq_len")
    window["pred_len"] = _as_positive_int(window.get("pred_len", 1), "window.pred_len")
    window["stride"] = _as_positive_int(window.get("stride", 1), "window.stride")

    # split
    split = cfg["split"]
    for f in ("train_ratio", "val_ratio", "test_ratio"):
        _assert(f in split, f"split.{f} 未设置")
        _assert(0 <= float(split[f]) <= 1, f"split.{f} 应位于 [0,1]")
    total = float(split["train_ratio"]) + float(split["val_ratio"]) + float(split["test_ratio"])
    _assert(abs(total - 1.0) < 1e-6 or total <= 1.0 + 1e-3, "split 三项之和应约等于 1")

    # model
    model = cfg["model"]
    _as_positive_int(model.get("hidden_dim", 64), "model.hidden_dim")
    _as_positive_int(model.get("num_layers", 2), "model.num_layers")
    dropout = float(model.get("dropout", 0.0))
    _assert(0.0 <= dropout < 1.0, "model.dropout 应位于 [0,1)")

    # train
    train = cfg["train"]
    train.setdefault("device", "cuda")
    train.setdefault("batch_size", 64)
    train.setdefault("epochs", 50)
    train.setdefault("lr", 1e-3)
    train.setdefault("patience", 10)
    train.setdefault("grad_clip", 1.0)
    train.setdefault("min_delta", 0.0)
    train.setdefault("seed", 42)

    # eval
    cfg["eval"].setdefault("metrics", ["mae", "mse", "rmse", "mape"]) 

    # predict
    cfg["predict"].setdefault("horizon", 30)

    # paths
    paths = cfg["paths"]
    paths.setdefault("outputs", "outputs")
    paths.setdefault("figures", "outputs/figures")
    for k in ("outputs", "figures"):
        paths[k] = str(Path(paths[k]))

    # run_name
    rn = cfg.get("run_name")
    if not rn or str(rn).strip() == "":
        # 延迟在主流程中补默认 run_name（含时间戳），以便命令行覆盖
        cfg["run_name"] = None

    return cfg

