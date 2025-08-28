# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

def ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Union[str, Path]):
    path = Path(path); ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_np(path: Union[str, Path], arr):
    path = Path(path); ensure_dir(path.parent)
    np.save(path, arr)

def save_csv(path: Union[str, Path], df: pd.DataFrame):
    path = Path(path); ensure_dir(path.parent)
    df.to_csv(path, index=False)

def snapshot_config(cfg: dict, out_dir: Union[str, Path]):
    ensure_dir(out_dir)
    save_json(cfg, Path(out_dir) / "config_snapshot.json")
