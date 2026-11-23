"""Build the project status dashboard (a single static HTML page) for GitHub Pages.

Every figure on the page is read from a file in the repository, so the page
cannot drift away from the report:

- energy_trading_mlops_proposal_v17.docx: Table 5.6 (test cases),
  Table 5.7 (defects), Table 5.8 (results; only the fixed-reward agents and
  Buy & Hold are shown), Table 5.10 (objectives)
- reports/evaluation_reward_fix/seed_results.csv and *_full_test_trajectory.csv:
  the five fixed-reward PPO-300k runs and Buy & Hold
- reports/live/: the closed loop - model_registry.json (registered versions,
  quality-gate results, champion, promotion and retrain history),
  monitor_status.json (champion health) and paper_trading.csv (paper
  trading, simulated, since 2026-07-07)
- dashboard/status.json: component status, latest DAG run, limitations
  (hand-maintained, mirrors Figure 4.5 and Chapter 5)
- git log: recent activity

Usage (from the repo root):  python dashboard/build_dashboard.py [out_dir]
Output: <out_dir>/index.html (default out_dir: site/)
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "energy_trading_mlops_proposal_v17.docx"
FIX_DIR = ROOT / "reports" / "evaluation_reward_fix"
LIVE_DIR = ROOT / "reports" / "live"
TEMPLATE = Path(__file__).with_name("template.html")
STATUS = Path(__file__).with_name("status.json")
REPO_URL = "https://github.com/GlynnCarson/energy-trading-mlops"


# ── Report tables ────────────────────────────────────────────────────────────
def report_tables(doc: Document) -> dict[str, list[list[str]]]:
    """Map 'Table 5.6' etc. to the rows of the table that follows its caption."""
    tables, caption = {}, ""
    for el in doc.element.body.iterchildren():
        if el.tag == qn("w:p"):
            text = "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
            if text:
                caption = text
        elif el.tag == qn("w:tbl"):
            m = re.match(r"(Table \d+\.\d+):", caption)
            if m and m.group(1) not in tables:
                rows = [[c.text.strip() for c in r.cells] for r in Table(el, doc).rows]
                tables[m.group(1)] = rows
    return tables


def verdict(text: str) -> str:
    """Final verdict of a test case: the last Pass/Fail word, or 'not run'."""
    if text.lower().startswith("not completed"):
        return "not_run"
    words = re.findall(r"\b(Pass|Fail)\b", text)
    return {"Pass": "pass", "Fail": "fail"}.get(words[-1] if words else "", "not_run")


def defect_state(text: str) -> str:
    for key, state in (("Fixed", "fixed"), ("Mitigated", "mitigated"), ("Open", "open")):
        if text.startswith(key):
            return state
    return "open"


def objective_state(text: str) -> str:
    return "partial" if text.lower().startswith("partially") else (
        "achieved" if text.lower().startswith("achieved") else "not_achieved")


# ── Evaluation data ──────────────────────────────────────────────────────────
def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trajectories() -> dict:
    s42 = read_csv(FIX_DIR / "PPO-300k-fixed-s42_full_test_trajectory.csv")
    out = {
        "dates": [r["date"] for r in s42],
        "price": [round(float(r["price"]), 3) for r in s42],
        # Buy & Hold from the fixed-reward evaluation (matches Table 5.8)
        "buy_hold": [round(float(r["buy_hold_value"]), 2) for r in s42],
        "seeds": {},
    }
    for seed in (42, 1, 2, 3, 4):
        rows = read_csv(FIX_DIR / f"PPO-300k-fixed-s{seed}_full_test_trajectory.csv")
        assert [r["date"] for r in rows] == out["dates"], f"seed {seed} dates differ"
        out["seeds"][str(seed)] = [round(float(r["agent_value"]), 2) for r in rows]
    return out


def seed_results() -> list[dict]:
    rows = []
    for r in read_csv(FIX_DIR / "seed_results.csv"):
        if int(r["steps"]) != 300_000:
            continue
        rows.append({
            "seed": int(r["seed"]),
            "val": float(r["val_final"]),
            "test252": float(r["test252"]),
            "test_full": float(r["test_full"]),
            "sharpe": float(r["sharpe"]),
            "max_dd": float(r["max_dd"]),
            "bh_full": float(r["bh_full"]),
            "bh252": float(r["bh252"]),
            "val_bh": float(r["val_bh"]),
        })
    return rows


def live() -> dict | None:
    """Closed loop: champion, gate/retrain history, monitor status and the paper-trading ledger."""
    if not (LIVE_DIR / "model_registry.json").exists():
        return None
    reg = json.loads((LIVE_DIR / "model_registry.json").read_text(encoding="utf-8"))
    mon = json.loads((LIVE_DIR / "monitor_status.json").read_text(encoding="utf-8"))
    rows = read_csv(LIVE_DIR / "paper_trading.csv")
    gate_initial = json.loads((LIVE_DIR / "gate_initial.json").read_text(encoding="utf-8"))
    paper = {
        "dates": [r["date"] for r in rows],
        "price": [float(r["price"]) for r in rows],
        "value": [float(r["portfolio_value"]) for r in rows],
        "buy_hold": [float(r["buy_hold_value"]) for r in rows],
        "position": [float(r["position_mmbtu"]) for r in rows],
        "version": [r["model_version"] for r in rows],
    }
    history = []
    events = reg.get("events", [])
    retro = [e for e in events if e["type"] == "gate" and e.get("context") == "retroactive"]
    if retro:   # one row for the retroactive gate on the July runs (details are in the gate table)
        n_pass = sum(e["passed"] for e in retro)
        history.append({"date": retro[0]["time"][:10], "kind": "Quality gate", "state": "pass" if n_pass else "fail",
                        "title": f"Retroactive gate on the {len(retro)} fixed-reward July runs: {n_pass} of {len(retro)} passed",
                        "detail": f"Validation {retro[0]['window']}; " + "; ".join(
                            f"seed {e['seed']} {e['val_return_pct']:+.2f} %" for e in retro)
                                  + ". Passing runs registered as versions of the model."})
    for e in events:
        t, date = e["type"], e["time"][:10]
        if t == "gate":
            if e.get("context") != "retroactive":
                history.append({"date": date, "kind": "Quality gate", "state": "pass" if e["passed"] else "fail",
                                "title": e.get("name", ""), "detail": e["reasons"]})
        elif t == "promotion":
            prev = f"replaced v{e['replaced']}" if e.get("replaced") else "first champion"
            history.append({"date": date, "kind": "Promotion", "state": "promoted",
                            "title": f"v{e['version']} became champion ({prev})", "detail": e["reason"]})
        elif t in ("retrain_required", "retrain_drill"):
            history.append({"date": date, "kind": "Monitor", "state": "drill" if t == "retrain_drill" else "retrain",
                            "title": "Retrain drill started" if t == "retrain_drill" else "Retrain required",
                            "detail": e["reason"] + (f" ({'; '.join(e['breaches'])})" if e.get("breaches") else "")})
        elif t == "retrain_aborted":
            history.append({"date": date, "kind": "Retrain (drill)", "state": "aborted",
                            "title": "Drill aborted, nothing registered or promoted", "detail": e["reason"]})
        elif t == "retrain":
            ch = "; ".join(f"seed {c['seed']} {'PASS' if c['passed'] else 'FAIL'} {c['val_return_pct']:+.2f} %"
                           for c in e["challengers"])
            champ = e.get("champion_val_return_pct")
            history.append({"date": date, "kind": "Retrain" + (" (smoke test)" if e.get("smoke_test")
                                                              else " (drill)" if e.get("drill") else ""),
                            "state": {"champion_replaced": "promoted", "failed_gate": "fail",
                                      "champion_kept": "kept"}[e["result"]],
                            "title": e["reason"],
                            "detail": f"Validation {e['val_window']}: {ch}"
                                      + (f"; champion v{e['champion_version']} {champ:+.2f} %" if champ is not None else "")})
    return {"registry": {k: reg.get(k) for k in ("model_name", "champion_alias", "champion", "versions",
                                                 "gate_thresholds", "exported_at")},
            "gate_initial": [{"seed": r["seed"], "run_id": r["run_id"], "passed": r["passed"],
                              "val": r["metrics"]["return_pct"], "bh": r["metrics"]["buy_and_hold_return_pct"],
                              "beaten": r["metrics"]["random_beaten_pct"], "dd": r["metrics"]["max_drawdown_pct"],
                              "window": f"{r['metrics']['window_start']} to {r['metrics']['window_end']}"}
                             for r in gate_initial["results"]],
            "monitor": mon, "paper": paper, "history": history}


def git_log(n: int = 12) -> list[dict]:
    try:
        out = subprocess.run(
            ["git", "log", f"-{n}", "--date=short", "--format=%ad%x09%h%x09%s"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [dict(zip(("date", "hash", "subject"), line.split("\t", 2)))
            for line in out.splitlines() if line.strip()]


# ── Build ────────────────────────────────────────────────────────────────────
def build(out_dir: Path) -> Path:
    tables = report_tables(Document(str(REPORT)))
    for needed in ("Table 5.6", "Table 5.7", "Table 5.8", "Table 5.10"):
        if needed not in tables:
            sys.exit(f"{needed} not found in {REPORT.name}")

    tests = [{"id": r[0], "req": r[1], "scenario": r[2], "result": r[3], "verdict": verdict(r[3])}
             for r in tables["Table 5.6"][1:]]
    defects = [{"id": r[0], "severity": r[1], "text": r[2], "status": r[3], "state": defect_state(r[3])}
               for r in tables["Table 5.7"][1:]]
    results = [dict(zip(("window", "strategy", "ret", "sharpe", "dd", "trades"), r))
               for r in tables["Table 5.8"][1:] if "cost-only" not in r[1]]
    objectives = [{"objective": r[0], "evidence": r[1], "outcome": r[2], "state": objective_state(r[2])}
                  for r in tables["Table 5.10"][1:]]

    commits = git_log()
    data = {
        "repo": REPO_URL,
        "report_url": f"{REPO_URL}/raw/main/{REPORT.name}",
        "last_commit": commits[0] if commits else None,
        "status": json.loads(STATUS.read_text(encoding="utf-8")),
        "tests": tests,
        "defects": defects,
        "results": results,
        "objectives": objectives,
        "seeds": seed_results(),
        "traj": trajectories(),
        "commits": commits,
        "live": live(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DASHBOARD_DATA__", payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site"
    print(f"Wrote {build(target)}")
