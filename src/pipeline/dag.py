"""
pipeline/dag.py
Apache Airflow DAG that orchestrates the full DataOps/MLOps pipeline:
  Ingest → Transform → Train DRL Agent → Backtest → Notify
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

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

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

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

    train_drl = PythonOperator(
        task_id="train_drl",
        python_callable=task_train_drl,
    )

    backtest_drl = PythonOperator(
        task_id="backtest_drl",
        python_callable=task_backtest_drl,
    )

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
    #          └── transform ──────── dq_report
    #                └── train_drl ── backtest_drl ── notify ── end

    start >> ingest >> transform
    transform >> dq_report
    transform >> train_drl >> backtest_drl >> notify_success >> end
    dq_report >> end
