import os
import argparse
import time
import json
import yaml
import optuna
import numpy as np
import pandas as pd

from train_eval import train_and_eval


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_list(s: str, cast=int):
    """把 '7,15,30' 解析成 [7, 15, 30]；空字符串/None 返回 None"""
    if s is None or str(s).strip() == "":
        return None
    return [cast(x.strip()) for x in str(s).split(",") if x.strip() != ""]


# ----------------- 构建 Optuna 目标函数 -----------------
def build_objective(
    base_cfg: dict,
    tune_root: str,
    direction_metric: str = "RMSE",
    # 搜索空间（None=使用默认集合）
    seq_lens=None,                 # e.g. [15]
    pred_lens=None,                # e.g. [1]
    hidden_dim_min=32,
    hidden_dim_max=256,
    hidden_dim_step=32,
    num_layers_min=1,
    num_layers_max=6,
    lr_min=1e-4,
    lr_max=1e-2,
    dropout_min=0.0,
    dropout_max=0.3,
):
    direction_metric = direction_metric.upper()
    assert direction_metric in {"RMSE", "MAE", "MAPE"}, "direction_metric 必须是 RMSE/MAE/MAPE 之一"

    def objective(trial: optuna.trial.Trial):
        cfg = dict(base_cfg)

        # —— 超参建议值 ——
        if seq_lens is None:
            cfg["seq_len"] = trial.suggest_categorical("seq_len", [15])
        else:
            cfg["seq_len"] = trial.suggest_categorical("seq_len", seq_lens)

        if pred_lens is None:
            cfg["pred_len"] = trial.suggest_categorical("pred_len", [1])
        else:
            cfg["pred_len"] = trial.suggest_categorical("pred_len", pred_lens)

        cfg["hidden_dim"]    = trial.suggest_int("hidden_dim", hidden_dim_min, hidden_dim_max, step=hidden_dim_step)
        cfg["num_layers"]    = trial.suggest_int("num_layers", num_layers_min, num_layers_max, step=1)
        cfg["learning_rate"] = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        cfg["dropout"]       = trial.suggest_float("dropout", dropout_min, dropout_max)

        run_name = f"optuna_trial_{trial.number:04d}"
        t0 = time.time()
        try:
            # ⚠️ 关键：调参不保存中间产物
            res = train_and_eval(cfg, run_name=run_name, save_outputs=False)

            test_avg = res["metrics"].get("TEST_AVG", None)
            if test_avg is None:
                over = res["metrics"].get("OVERALL", None)
                if over is None:
                    raise RuntimeError("未找到 TEST_AVG 或 OVERALL 指标，请更新 train_and_eval 返回值。")
                metric_block = over
            else:
                metric_block = test_avg["OVERALL"]

            rmse = float(metric_block["RMSE"])
            mae  = float(metric_block["MAE"])
            mape = float(metric_block["MAPE"])

            trial.set_user_attr("metrics_block", metric_block)
            obj = {"RMSE": rmse, "MAE": mae, "MAPE": mape}[direction_metric]
        except Exception as e:
            trial.set_user_attr("error", str(e))
            obj = float("inf")

        elapsed = time.time() - t0
        trial.set_user_attr("time_sec", elapsed)
        print(f"[Trial #{trial.number:04d}] seq_len={cfg['seq_len']}, pred_len={cfg['pred_len']}, "
              f"h={cfg['hidden_dim']}, L={cfg['num_layers']}, lr={cfg['learning_rate']:.2e}, dp={cfg['dropout']:.2f} "
              f"=> {direction_metric}={obj:.6f} | time={elapsed:.1f}s")
        return obj

    return objective


# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="基础配置文件（在此基础上搜索）")
    ap.add_argument("--metric", type=str, default="RMSE", choices=["RMSE","MAE","MAPE"], help="优化目标")
    ap.add_argument("--n_trials", type=int, default=20, help="搜索次数")
    ap.add_argument("--storage", type=str, default=None,
                    help="optuna storage，例如 sqlite:///tuning.db；若设定且配合 --study_name 可断点续跑")
    ap.add_argument("--study_name", type=str, default=None, help="study 名称；配合 --storage 可续跑")

    # 搜索空间（可选；留空走默认）
    ap.add_argument("--seq_lens", type=str, default="", help="逗号分隔，如 15；留空默认[15]")
    ap.add_argument("--pred_lens", type=str, default="", help="逗号分隔，如 1；留空默认[1]")
    ap.add_argument("--hidden_dim_min", type=int, default=32)
    ap.add_argument("--hidden_dim_max", type=int, default=256)
    ap.add_argument("--hidden_dim_step", type=int, default=32)
    ap.add_argument("--num_layers_min", type=int, default=1)
    ap.add_argument("--num_layers_max", type=int, default=6)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_max", type=float, default=1e-2)
    ap.add_argument("--dropout_min", type=float, default=0.0)
    ap.add_argument("--dropout_max", type=float, default=0.3)

    args = ap.parse_args()

    base_cfg = load_cfg(args.config)

    # 结果输出根目录（只存最终汇总文件）
    ts = time.strftime("%Y%m%d_%H%M%S")
    tune_root = os.path.join(base_cfg.get("outputs_dir", "./outputs"), "tuning", ts)
    ensure_dir(tune_root)

    # 解析列表参数
    seq_lens = parse_list(args.seq_lens, int)
    pred_lens = parse_list(args.pred_lens, int)

    # 创建 study
    sampler = optuna.samplers.TPESampler(seed=base_cfg.get("seed", 42))
    pruner  = optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage and args.study_name),
    )

    objective = build_objective(
        base_cfg=base_cfg,
        tune_root=tune_root,
        direction_metric=args.metric,
        seq_lens=seq_lens,
        pred_lens=pred_lens,
        hidden_dim_min=args.hidden_dim_min,
        hidden_dim_max=args.hidden_dim_max,
        hidden_dim_step=args.hidden_dim_step,
        num_layers_min=args.num_layers_min,
        num_layers_max=args.num_layers_max,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
    )

    start_all = time.time()
    per_trial_times = []

    def _callback(study_, trial):
        if "time_sec" in trial.user_attrs:
            per_trial_times.append(float(trial.user_attrs["time_sec"]))
        done = len(per_trial_times)
        remaining = args.n_trials - done
        avg = np.mean(per_trial_times) if per_trial_times else 0.0
        eta_sec = remaining * avg
        print(f"[Progress] {done}/{args.n_trials} finished | avg={avg:.1f}s/trial | ETA~{eta_sec/60:.1f} min\n")

    study.optimize(objective, n_trials=args.n_trials, callbacks=[_callback])

    # 仅保存最终结果
    best_params = study.best_params
    best_value  = study.best_value

    print("\n===== 搜索完成 =====")
    print("最佳参数：", best_params)
    print(f"最佳 {args.metric.upper()}: {best_value:.6f}")

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df_csv = os.path.join(tune_root, "optuna_trials.csv")
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")

    best_cfg = dict(base_cfg)
    for k, v in best_params.items():
        best_cfg[k] = v
    best_cfg_path = os.path.join(tune_root, "best_config.yaml")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    summary = {
        "metric": args.metric.upper(),
        "best_value": float(best_value),
        "best_params": best_params,
        "n_trials": args.n_trials,
        "storage": args.storage,
        "study_name": args.study_name,
        "tune_root": tune_root,
        "total_minutes": (time.time() - start_all) / 60.0
    }
    save_json(summary, os.path.join(tune_root, "summary.json"))

    print("\n文件已保存：")
    print("  Trials CSV ->", df_csv)
    print("  Best CFG   ->", best_cfg_path)
    print("  Summary    ->", os.path.join(tune_root, "summary.json"))
    print(f"总耗时      -> {summary['total_minutes']:.1f} 分钟")


if __name__ == "__main__":
    main()
