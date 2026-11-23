"""
Regression tests for the orchestration defects found when the DAG was run in Docker Compose (Nov 2026).

Airflow is not installed on the development workstation (it runs in the Docker Compose stack), so the DAG test
inspects dag.py's source instead of importing it.
"""
import ast
from pathlib import Path

from src.ml.drl.train_drl import load_drl_config

DAG_FILE = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "dag.py"


def test_dag_end_task_fails_when_an_upstream_task_fails():
    """DEF-12: with trigger_rule=ALL_DONE the run was marked successful even when train_drl failed.
    `end` is the only leaf task, so it must keep the default ALL_SUCCESS rule."""
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    end_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(k.arg == "task_id" and isinstance(k.value, ast.Constant) and k.value.value == "end"
                for k in node.keywords)
    ]
    assert len(end_calls) == 1, "expected exactly one task with task_id='end'"
    rules = [ast.unparse(k.value) for k in end_calls[0].keywords if k.arg == "trigger_rule"]
    assert not rules or rules[0] in ("TriggerRule.ALL_SUCCESS", "'all_success'"), rules


def test_default_drl_config_matches_the_evaluated_daily_agents():
    """DEF-13: the DAG trains with the drl.yaml defaults, which must be the daily EIA settings of the
    Chapter 5 PPO-300k agents (train_on_eia.py / retrain_fixed_reward.py), not the old hourly ones."""
    cfg = load_drl_config()
    env, tr, bt = cfg["environment"], cfg["training"], cfg["backtest"]
    assert env["episode_length"] == 252
    assert env["max_position_mwh"] == 25_000
    assert env["stop_loss_pct"] == 0.50
    assert env["transaction_cost_pct"] == 0.001
    assert tr["algorithm"] == "PPO" and tr["total_timesteps"] == 300_000 and tr["n_envs"] == 4
    assert bt["periods_per_year"] == 252 and bt["steps_per_day"] == 1
    assert bt["units"]["price"] == "$/MMBtu"
