# -*- coding: utf-8 -*-
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.io import ensure_dir, snapshot_config

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, best_is_min=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_is_min = best_is_min
        self.counter = 0
        self.best = np.inf if best_is_min else -np.inf
        self.stop = False

    def _is_better(self, score):
        if self.best_is_min:
            return score < self.best - self.min_delta
        else:
            return score > self.best + self.min_delta

    def step(self, score):
        if self._is_better(score):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

def train_loop(model, train_ds, val_ds, cfg: dict, out_dir, test_size=None):
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    print(
        "Training configuration:\n"
        f"  device: {device}\n"
        f"  epochs: {cfg['train']['epochs']}\n"
        f"  batch_size: {cfg['train']['batch_size']}\n"
        f"  learning_rate: {cfg['train']['lr']}"
    )
    print(
        f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, "
        f"test: {test_size if test_size is not None else 'N/A'}"
    )

    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.MSELoss()

    # 学习率调度（按验证集loss降不动而衰减）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=max(1, cfg["train"]["patience"] // 2),
        min_lr=1e-6,
    )
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    ckpt_dir = Path(out_dir) / "checkpoints"
    ensure_dir(ckpt_dir)
    snapshot_config(cfg, Path(out_dir) / "logs")

    early = EarlyStopper(
        patience=cfg["train"]["patience"],
        min_delta=float(cfg["train"].get("min_delta", 0.0)),
        best_is_min=True,
    )
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        epoch_start = time.time()
        model.train(); train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * len(x)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (cfg["train"]["epochs"] - epoch)
        print(
            f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"lr {optim.param_groups[0]['lr']:.2e} | time {epoch_time:.2f}s | "
            f"eta {remaining/60:.2f}m"
        )

        # 保存最佳
        if val_loss <= early.best:
            torch.save({"model": model.state_dict()}, best_path)
        early.step(val_loss)
        if early.stop:
            print("Early stopping triggered.")
            break

    print("Best model saved at:", best_path)
    return best_path
