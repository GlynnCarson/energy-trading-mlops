"""
src/ml/drl/closed_loop_tasks.py
─────────────────────────────────────────────────────────────────────────────
Airflow callables for the closed loop:

  paper_trade -> monitor -> branch_on_health
        ├── skip_retrain
        └── alert_retrain_required -> prepare_retrain -> train_challenger_s{seed} (one task per seed)
                -> evaluate_and_promote -> branch_on_outcome
                      ├── alert_champion_replaced
                      ├── alert_retrain_failed_gate   (no challenger passed the gate)
                      └── alert_champion_kept         (passed the gate, did not beat the champion)

A retrain drill is started with:  airflow dags trigger ... -c '{"force_retrain": true}'
Paper trading is simulated: no real money is involved.
"""

from __future__ import annotations


def _conf(context) -> dict:
    return getattr(context.get("dag_run"), "conf", None) or {}


def task_paper_trade(**context):
    from src.ml.drl.paper_trade import update
    summary = update()
    context["ti"].xcom_push(key="paper_summary", value=summary)
    return summary


def task_monitor(**context):
    from src.ml.drl import registry
    from src.ml.drl.monitor import run_monitor
    from src.ml.drl.train_drl import load_drl_config
    cfg = load_drl_config()
    status = run_monitor(cfg, force=bool(_conf(context).get("force_retrain", False)))
    registry.export_registry(cfg)
    slim = {k: status[k] for k in ("status", "status_label", "days", "metrics", "breaches", "retrain",
                                   "champion_version", "last_data_date")}
    context["ti"].xcom_push(key="monitor", value=slim)
    return slim


def branch_on_health(**context):
    m = context["ti"].xcom_pull(task_ids="monitor", key="monitor")
    return "alert_retrain_required" if m["retrain"]["triggered"] else "skip_retrain"


def task_prepare_retrain(**context):
    from src.ml.drl.retrain import prepare
    meta = prepare(drill=bool(_conf(context).get("force_retrain", False)))
    context["ti"].xcom_push(key="retrain_id", value=meta["retrain_id"])
    return meta


def make_train_challenger(seed: int):
    def _train(**context):
        from src.ml.drl.retrain import train
        rid = context["ti"].xcom_pull(task_ids="prepare_retrain", key="retrain_id")
        run_id = train(rid, seed)
        context["ti"].xcom_push(key="run_id", value=run_id)
        return run_id
    return _train


def make_evaluate_and_promote(seeds: list[int]):
    def _evaluate(**context):
        from src.ml.drl.retrain import evaluate_and_promote
        ti = context["ti"]
        rid = ti.xcom_pull(task_ids="prepare_retrain", key="retrain_id")
        ids = {s: ti.xcom_pull(task_ids=f"train_challenger_s{s}", key="run_id") for s in seeds}
        out = evaluate_and_promote(rid, ids)
        slim = {k: out[k] for k in ("retrain_id", "drill", "result", "reason", "promoted",
                                    "champion_version", "new_champion_version", "val_window")}
        slim["challengers"] = [{k: c.get(k) for k in ("seed", "passed", "val_return_pct", "reasons", "version")}
                               for c in out["challengers"]]
        slim["champion_val_return_pct"] = out["champion_val_return_pct"]
        ti.xcom_push(key="outcome", value=slim)
        return slim
    return _evaluate


def branch_on_outcome(**context):
    o = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="outcome")
    return {"champion_replaced": "alert_champion_replaced", "failed_gate": "alert_retrain_failed_gate",
            "champion_kept": "alert_champion_kept"}[o["result"]]
