# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "lstm_forecast", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    创建既输出到控制台又可选输出到文件的标准 logger。
    - name: 日志器名称
    - log_file: 若提供，将把日志写入该文件，并创建父目录
    - level: 日志级别，默认 INFO
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # 已初始化（避免重复添加 handler）
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

