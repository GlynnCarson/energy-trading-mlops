"""
pipeline/dag.py
Apache Airflow DAG that orchestrates the full DataOps/MLOps pipeline:
  Ingest → Transform → Train DRL Agent → Backtest → Notify
and the closed loop (paper trading is simulated, no real money):
  Ingest → Paper trade (champion) → Monitor → retrain required?
      yes → Retrain challengers (one task per seed) → Quality gate + promote → alert
      no  → skip
A retrain drill: airflow dags trigger energy_trading_mlops_pipeline -c '{"force_retrain": true}'
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator

# ── Default DAG args ──────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops-alerts@energytrading.io"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="energy_trading_mlops_pipeline",
    default_args=default_args,
    description="End-to-end DataOps/MLOps pipeline for energy trading via DRL",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energy", "trading", "mlops", "dataops", "drl"],
) as dag:

    # ──────────────────────────────────────────
    # Task functions
    # ──────────────────────────────────────────

    def task_ingest(**context):
        """Pull raw energy data and land on S3."""
        from src.ingestion.ingest import run_ingestion
        execution_date = context["ds"]
        s3_path = run_ingestion(execution_date=execution_date)
        context["ti"].xcom_push(key="raw_s3_path", value=s3_path)
        return s3_path

    def task_transform(**context):
        """Clean, validate, and feature-engineer the raw data."""
        from src.transformation.transform import run_transformation
        raw_path = context["ti"].xcom_pull(task_ids="ingest", key="raw_s3_path")
        split_paths = run_transformation(raw_path)
        context["ti"].xcom_push(key="split_paths", value=split_paths)
        return split_paths

    def task_data_quality_report(**context):
        """Generate and log a data quality summary for the processed dataset."""
        from src.utils.helpers import load_pipeline_config, download_df_from_s3
        split_paths = context["ti"].xcom_pull(task_ids="transform", key="split_paths")
        cfg = load_pipeline_config()
        bucket = cfg["aws"]["s3_bucket"]

        _, _, bk = split_paths["train"].partition("s3://")
        b, _, k = bk.partition("/")
        import boto3, json
        df = download_df_from_s3(b, k)

        report = {
            "train_rows": len(df),
            "n_features": len(df.columns),
            "null_pct": df.isnull().mean().to_dict(),
            "price_stats": df[cfg["transformation"]["target_column"]].describe().to_dict(),
        }
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"reports/dq/{context['ds']}.json",
            Body=json.dumps(report, default=str),
        )
        return report

    # ──────────────────────────────────────────
    # Task definitions
    # ──────────────────────────────────────────

    start = EmptyOperator(task_id="start")
    # Default trigger rule (ALL_SUCCESS): `end` is the DAG's only leaf, so if any task fails, `end` becomes
    # upstream_failed and Airflow marks the whole run failed. With ALL_DONE a run whose training failed was
    # still shown as successful (DEF-12).
    end = EmptyOperator(task_id="end")

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_transform,
    )

    dq_report = PythonOperator(
        task_id="data_quality_report",
        python_callable=task_data_quality_report,
    )

    from src.ml.drl.drl_tasks import task_train_drl, task_backtest_drl

    # Every PPO training task uses the 1-slot "training" pool (created by airflow-init), so trainings run one
    # at a time: four in parallel on one machine oversubscribed the CPU and each ran ~20x slower.
    train_drl = PythonOperator(
        task_id="train_drl",
        python_callable=task_train_drl,
        pool="training",
    )

    backtest_drl = PythonOperator(
        task_id="backtest_drl",
        python_callable=task_backtest_drl,
    )

    # ── Closed loop: paper trading (simulated) → monitor → retrain / gate / promote ──
    import yaml
    from pathlib import Path
    import src as _project_src
    from src.ml.drl import closed_loop_tasks as cl

    _cfg_file = Path(_project_src.__file__).resolve().parents[1] / "config" / "drl.yaml"
    RETRAIN_SEEDS = yaml.safe_load(_cfg_file.read_text())["retraining"]["seeds"]

    paper_trade = PythonOperator(task_id="paper_trade", python_callable=cl.task_paper_trade)
    monitor = PythonOperator(task_id="monitor", python_callable=cl.task_monitor)
    branch_health = BranchPythonOperator(task_id="branch_on_health", python_callable=cl.branch_on_health)
    skip_retrain = EmptyOperator(task_id="skip_retrain")

    alert_retrain_required = EmailOperator(
        task_id="alert_retrain_required",
        to=["team@energytrading.io"],
        subject=(
            "[Energy MLOps] {{ 'Retrain drill started' if dag_run.conf.get('force_retrain') else 'Retrain required' }}"
            " - champion v{{ ti.xcom_pull(task_ids='monitor', key='monitor')['champion_version'] }} - {{ ds }}"
        ),
        html_content="""
            <h3>Retrain triggered (paper trading is simulated)</h3>
            {% set m = ti.xcom_pull(task_ids='monitor', key='monitor') %}
            <p><b>Champion:</b> version {{ m['champion_version'] }} | <b>Health:</b> {{ m['status_label'] }}
               over {{ m['days'] }} trading days (data to {{ m['last_data_date'] }})</p>
            <p><b>Reason:</b> {{ m['retrain']['reason'] }}</p>
            <p><b>Breaches:</b> {{ m['breaches'] or 'none' }}</p>
            <p><b>Metrics:</b> {{ m['metrics'] }}</p>
        """,
    )
    prepare_retrain = PythonOperator(task_id="prepare_retrain", python_callable=cl.task_prepare_retrain)
    train_challengers = [
        PythonOperator(task_id=f"train_challenger_s{s}", python_callable=cl.make_train_challenger(s),
                       pool="training")
        for s in RETRAIN_SEEDS
    ]
    evaluate_promote = PythonOperator(task_id="evaluate_and_promote",
                                      python_callable=cl.make_evaluate_and_promote(RETRAIN_SEEDS))
    branch_outcome = BranchPythonOperator(task_id="branch_on_outcome", python_callable=cl.branch_on_outcome)

    _outcome_body = """
        {% set o = ti.xcom_pull(task_ids='evaluate_and_promote', key='outcome') %}
        <p><b>Retrain:</b> {{ o['retrain_id'] }}{{ ' (smoke test: nothing registered or promoted)' if o['smoke_test']
           else (' (drill)' if o['drill']) }} |
           <b>validation window:</b> {{ o['val_window'] }}</p>
        <p><b>Decision:</b> {{ o['reason'] }}</p>
        <p><b>Champion on the same window:</b> {{ o['champion_val_return_pct'] }} %</p>
        <ul>{% for c in o['challengers'] %}<li>seed {{ c['seed'] }}: {{ 'PASS' if c['passed'] else 'FAIL' }},
            validation {{ '%+.2f' % c['val_return_pct'] }} % - {{ c['reasons'] }}</li>{% endfor %}</ul>
    """
    alert_replaced = EmailOperator(
        task_id="alert_champion_replaced",
        to=["team@energytrading.io"],
        subject=(
            "[Energy MLOps] Champion replaced: "
            "v{{ ti.xcom_pull(task_ids='evaluate_and_promote', key='outcome')['new_champion_version'] }} - {{ ds }}"
        ),
        html_content="<h3>New champion promoted</h3>" + _outcome_body,
    )
    alert_failed_gate = EmailOperator(
        task_id="alert_retrain_failed_gate",
        to=["team@energytrading.io"],
        subject="[Energy MLOps] Retrain failed the quality gate - champion kept - {{ ds }}",
        html_content="<h3>No challenger passed the quality gate</h3>" + _outcome_body,
    )
    alert_kept = EmailOperator(
        task_id="alert_champion_kept",
        to=["team@energytrading.io"],
        subject="[Energy MLOps] Retrain did not beat the champion - champion kept - {{ ds }}",
        html_content="<h3>Challengers passed the gate but did not beat the champion</h3>" + _outcome_body,
    )
    # Joins the branches. NONE_FAILED_MIN_ONE_SUCCESS lets a skipped branch through but still turns
    # any failure into upstream_failed, so `end` (ALL_SUCCESS, DEF-12) still fails the run.
    closed_loop_done = EmptyOperator(task_id="closed_loop_done",
                                     trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    notify_success = EmailOperator(
        task_id="notify_success",
        to=["team@energytrading.io"],
        subject="[Energy MLOps] Pipeline Succeeded — {{ ds }}",
        html_content="""
            <h3>✅ Energy Trading MLOps Pipeline Completed</h3>
            <p><b>Date:</b> {{ ds }}</p>
            <p><b>DRL MLflow Run ID:</b> {{ ti.xcom_pull(task_ids='train_drl', key='drl_run_id') }}</p>
            <p><b>Backtest Summary:</b> {{ ti.xcom_pull(task_ids='backtest_drl', key='drl_backtest_summary') }}</p>
        """,
    )

    # ──────────────────────────────────────────
    # DAG wiring
    # ──────────────────────────────────────────
    #
    #  start
    #    └── ingest
    #          ├── transform ── dq_report ───────────────────────────────────────── end
    #          │     └── train_drl ── backtest_drl ── notify ───────────────────── end
    #          └── paper_trade ── monitor ── branch_on_health
    #                                          ├── skip_retrain ─────────────────── closed_loop_done ── end
    #                                          └── alert_retrain_required
    #                                                └── prepare_retrain
    #                                                      └── train_challenger_s<seed> (one per seed)
    #                                                            └── evaluate_and_promote
    #                                                                  └── branch_on_outcome
    #                                                                        └── alert_* ── closed_loop_done

    start >> ingest >> transform
    transform >> dq_report
    transform >> train_drl >> backtest_drl >> notify_success >> end
    dq_report >> end

    ingest >> paper_trade >> monitor >> branch_health
    branch_health >> skip_retrain >> closed_loop_done
    branch_health >> alert_retrain_required >> prepare_retrain >> train_challengers >> evaluate_promote
    evaluate_promote >> branch_outcome >> [alert_replaced, alert_failed_gate, alert_kept]
    [alert_replaced, alert_failed_gate, alert_kept] >> closed_loop_done
    closed_loop_done >> end
