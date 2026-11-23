"""
src/ml/drl/drl_tasks.py
─────────────────────────────────────────────────────────────────────────────
Airflow-compatible task functions that plug the DRL training + backtest
into the energy_trading_mlops_pipeline DAG:
    ... → transform → train_drl → backtest_drl → notify
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Airflow task: train DRL agent
# ─────────────────────────────────────────────────────────────────────────────

def task_train_drl(**context):
    """
    Airflow PythonOperator callable.
    Trains the DRL agent and pushes (run_id, model_path) to XCom.
    """
    from src.ml.drl.train_drl import run_drl_training, load_drl_config
    from src.utils.helpers import download_df_from_s3

    split_paths = context["ti"].xcom_pull(task_ids="transform", key="split_paths")

    def load(path):
        _, _, bk = path.partition("s3://")
        b, _, k = bk.partition("/")
        return download_df_from_s3(b, k)

    train_df = load(split_paths["train"])
    val_df = load(split_paths["val"])

    drl_cfg = load_drl_config()

    # Optional override from the DAG-run config, e.g. a short demo run:
    #   airflow dags trigger energy_trading_mlops_pipeline -c '{"total_timesteps": 20000}'
    conf = getattr(context.get("dag_run"), "conf", None) or {}
    if "total_timesteps" in conf:
        drl_cfg["training"]["total_timesteps"] = int(conf["total_timesteps"])
        logger.info(f"[DRL] total_timesteps overridden by DAG-run conf: {conf['total_timesteps']}")

    run_id = run_drl_training(train_df, val_df, drl_cfg)

    context["ti"].xcom_push(key="drl_run_id", value=run_id)
    context["ti"].xcom_push(key="drl_cfg", value=drl_cfg)
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Airflow task: backtest DRL agent
# ─────────────────────────────────────────────────────────────────────────────

def task_backtest_drl(**context):
    """
    Airflow PythonOperator callable.
    Loads best checkpoint, runs full backtest, uploads report to S3.
    """
    import boto3
    from pathlib import Path
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from src.ml.drl.backtest import run_full_backtest
    from src.ml.drl.train_drl import load_drl_config
    from src.utils.helpers import download_df_from_s3, load_pipeline_config

    drl_cfg = context["ti"].xcom_pull(task_ids="train_drl", key="drl_cfg") or load_drl_config()
    split_paths = context["ti"].xcom_pull(task_ids="transform", key="split_paths")

    def load(path):
        _, _, bk = path.partition("s3://")
        b, _, k = bk.partition("/")
        return download_df_from_s3(b, k)

    test_df = load(split_paths["test"])

    # Load best model from checkpoint
    algo = drl_cfg["training"]["algorithm"]
    best_path = Path(drl_cfg["training"]["best_model_path"]) / "best_model"
    AlgoMap = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
    agent = AlgoMap[algo].load(str(best_path))

    output_dir = drl_cfg.get("backtest", {}).get("output_dir", "./reports/backtest")
    summary = run_full_backtest(agent, test_df, drl_cfg, output_dir=output_dir)

    # Upload report to S3
    pipe_cfg = load_pipeline_config()
    bucket = pipe_cfg["aws"]["s3_bucket"]
    s3 = boto3.client("s3")
    report_path = summary["report_path"]
    if report_path:
        s3_key = f"reports/drl/{context['ds']}/backtest_report.png"
        s3.upload_file(report_path, bucket, s3_key)
        logger.info(f"[DRL] Backtest report → s3://{bucket}/{s3_key}")

    context["ti"].xcom_push(key="drl_backtest_summary", value=summary["agent"])
    return summary["agent"]
