"""
demo/app.py
─────────────────────────────────────────────────────────────────────────────
Local web app for the defense demo of the live model:
PAPER TRADING (SIMULATED) on new EIA prices.

    python demo/app.py          # opens http://127.0.0.1:8050 in the browser

  ▶ Trade new days  fetches the latest EIA prices and lets the champion model
                    trade every day published since the demo ledger's last day,
                    using the real paper-trading code (src/ml/drl/paper_trade.py):
                    features for day t from data up to t only, the model loaded
                    from the MLflow registry, the same costs, limits and stop-loss.
                    The page then animates the new days one by one. Pressing it
                    again with no new prices trades nothing (no day is traded twice).
  ↺ Reset demo      puts the demo ledger back to its frozen starting point.

The demo trades into its own copy of the ledger in reports/demo/, so the real
record in reports/live/ (used by the dashboard and the Airflow DAG) is never
changed by this app.
  reports/demo/baseline/      frozen copy of the ledger the demo starts from
  reports/demo/workspace/     the ledger the demo trades into (reset copies baseline here)
  reports/demo/eia_cache.csv  last EIA data fetched; used only if the API is down
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    print("Starting the demo (loading PyTorch and MLflow takes about 20 seconds)…", flush=True)

from src.ml.drl import paper_trade  # noqa: E402
from src.ml.drl.monitor import assess  # noqa: E402
from src.ml.drl.train_drl import load_drl_config  # noqa: E402

LIVE = ROOT / "reports" / "live"
DEMO = ROOT / "reports" / "demo"
BASELINE, WORKSPACE = DEMO / "baseline", DEMO / "workspace"
CACHE, CACHE_META = DEMO / "eia_cache.csv", DEMO / "eia_cache.json"
FILES = ("paper_trading.csv", "paper_state.json")
PORT = 8050

# The paper-trading update writes to these module paths: point them at the demo copy.
paper_trade.LEDGER = WORKSPACE / "paper_trading.csv"
paper_trade.STATE = WORKSPACE / "paper_state.json"
CFG = load_drl_config()


# ── Demo files ───────────────────────────────────────────────────────────────
def reset_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(BASELINE / f, WORKSPACE / f)


def ensure_demo_files() -> None:
    """Freeze the baseline from the real ledger the first time; make sure a workspace exists."""
    if not all((BASELINE / f).exists() for f in FILES):
        BASELINE.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy2(LIVE / f, BASELINE / f)
    if not all((WORKSPACE / f).exists() for f in FILES):
        reset_workspace()


def read_ledger() -> pd.DataFrame:
    return pd.read_csv(WORKSPACE / "paper_trading.csv", dtype={"model_version": str, "seed": str})


def health(df: pd.DataFrame) -> dict:
    h = assess(df, CFG["monitoring"])
    m = h["metrics"]
    return {"status": h["status"], "days": h["days"], "ret": m["return_since_deploy_pct"], "bh": m["buy_hold_since_deploy_pct"],
            "gap": m["gap_since_deploy_pp"], "dd": m["drawdown_from_peak_pct"]}


def rows(df: pd.DataFrame) -> list[dict]:
    return [{"date": r.date, "price": float(r.price), "pos": float(r.position_mmbtu), "value": float(r.portfolio_value),
             "bh": float(r.buy_hold_value), "version": str(r.model_version)} for r in df.itertuples()]


def state_payload() -> dict:
    df = read_ledger()
    base = pd.read_csv(BASELINE / "paper_trading.csv")
    bh = json.loads((WORKSPACE / "paper_state.json").read_text())["buy_hold"]
    return {"rows": rows(df), "health": health(df), "baseline_last": base["date"].iloc[-1], "cache": cache_info(),
            "bh_hold": {"mmbtu": bh["position_mwh"], "price": bh["avg_entry_price"], "first_day": df["date"].iloc[0]},
            "thresholds": {k: CFG["monitoring"][k] for k in ("max_gap_since_deploy_pp", "max_drawdown_from_peak_pct")}}


def cache_info() -> dict | None:
    """What the saved EIA copy holds, for the offline-mode switch."""
    if not CACHE.exists():
        return None
    meta = json.loads(CACHE_META.read_text()) if CACHE_META.exists() else {}
    last = pd.read_csv(CACHE, usecols=[0]).iloc[-1, 0]
    return {"saved_at": meta.get("fetched_at", "unknown"), "source": meta.get("source", "EIA"), "last_date": str(last)[:10]}


def save_cache(raw: pd.DataFrame, source: str) -> str:
    DEMO.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CACHE)
    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    CACHE_META.write_text(json.dumps({"fetched_at": stamp, "source": source}), encoding="utf-8")
    return stamp


def load_cache(reason: str) -> tuple[pd.DataFrame, str]:
    info = cache_info()
    if info is None:
        raise RuntimeError("there is no saved EIA copy yet: run the app once while EIA is reachable, or `python demo/app.py --prefetch`")
    log(f"{reason} Using the copy saved at {info['saved_at']} ({info['source']}, prices up to {info['last_date']}).")
    raw = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    return raw, f"the saved copy from {info['saved_at']}, downloaded from the {info['source']}"


# ── Trading job (background thread, polled by the page) ──────────────────────
JOB = {"running": False, "messages": [], "result": None, "error": None}
LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOCK:
        JOB["messages"].append(msg)


# EIA's published download files: the same daily series as the API, served as static files,
# which keep working when the API answers 504 Gateway Timeout. The API also carries about
# 20 extra rows since 2010 (US market holidays) that the files leave out, so the API stays first.
EIA_FILES = {"price_usd": "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls",       # Henry Hub gas, $/MMBtu
             "oil_price_usd": "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"}     # WTI crude, $/bbl


def fetch_eia_files(start: str = "2010-01-01") -> pd.DataFrame:
    """Same layout as EIAIngester.build_dataset (gas target joined with crude, forward-filled)."""
    import io

    import requests

    cols = {}
    for col, url in EIA_FILES.items():
        r = requests.get(url, timeout=60, headers={"User-Agent": "energy-trading-mlops demo"})
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
        df.columns = ["timestamp", col]
        df = df.dropna()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cols[col] = df.set_index("timestamp")[col].astype(float)
    out = cols["price_usd"].to_frame().join(cols["oil_price_usd"], how="left").ffill().dropna()
    out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    out["data_source"] = "eia_file"
    return out


def fetch_prices(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """EIA API first (what the pipeline uses), then EIA's download files, then the saved copy.
    Offline mode goes straight to the saved copy."""
    if offline:
        return load_cache("Offline mode: nothing is downloaded.")
    from dotenv import load_dotenv

    from src.ingestion import ingest
    from src.ingestion.ingest import EIAIngester

    load_dotenv(ROOT / ".env")
    ingest.ing["timeout_seconds"] = 30   # one try: a good download takes about 20 s, a failing one about 60 s
    ingest.ing["retry_attempts"] = 1
    try:
        log("Fetching the latest Henry Hub gas and WTI crude prices from the EIA API (up to 30 seconds)…")
        raw = EIAIngester().build_dataset(start="2010-01-01", commodity="gas")
        stamp = save_cache(raw, "EIA API")
        return raw, f"the EIA API (fetched live at {stamp})"
    except Exception:  # EIA often answers 504 Gateway Timeout
        pass
    try:
        log("The EIA API did not respond; downloading EIA's published price files instead…")
        raw = fetch_eia_files()
        stamp = save_cache(raw, "EIA download files")
        return raw, f"EIA's published price files, fetched live at {stamp} because the API did not respond"
    except Exception as exc:
        if cache_info() is None:
            raise RuntimeError(f"EIA could not be reached ({type(exc).__name__}) and there is no saved copy yet") from exc
        raw, source = load_cache("EIA could not be reached.")
        return raw, source


def trade_job(offline: bool) -> None:
    try:
        before = len(read_ledger())
        raw, source = fetch_prices(offline)
        log(f"EIA prices up to {raw.index[-1]:%d %b %Y}. Loading the champion from the MLflow registry and trading each new day…")
        summary = paper_trade.update(CFG, raw)
        df = read_ledger()
        result = {"summary": summary, "source": source, "rows": rows(df), "new_from": before,
                  "steps": [health(df.iloc[:k]) for k in range(before + 1, len(df) + 1)]}
        with LOCK:
            JOB["result"] = result
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            JOB["error"] = str(exc)
    finally:
        with LOCK:
            JOB["running"] = False


# ── Web app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/api/state")
def api_state():
    return jsonify(state_payload())


@app.get("/api/test")
def api_test():
    """Held-out test period (Chapter 5): the five fixed-reward PPO-300k seeds and Buy & Hold, 414 trading days."""
    d = ROOT / "reports" / "evaluation_reward_fix"
    seeds, out = (42, 1, 2, 3, 4), {}
    for sd in seeds:
        t = pd.read_csv(d / f"PPO-300k-fixed-s{sd}_full_test_trajectory.csv")
        out[str(sd)] = [round(float(v), 2) for v in t["agent_value"]]
    t = pd.read_csv(d / "PPO-300k-fixed-s42_full_test_trajectory.csv")
    champ = str(json.loads((LIVE / "model_registry.json").read_text())["champion"]["seed"])
    return jsonify({"dates": t["date"].tolist(), "bh": [round(float(v), 2) for v in t["buy_hold_value"]], "seeds": out, "champion": champ})


@app.post("/api/trade")
def api_trade():
    with LOCK:
        if JOB["running"]:
            return jsonify({"started": False})
        JOB.update(running=True, messages=[], result=None, error=None)
    offline = bool((request.get_json(silent=True) or {}).get("offline"))
    threading.Thread(target=trade_job, args=(offline,), daemon=True).start()
    return jsonify({"started": True})


@app.get("/api/job")
def api_job():
    with LOCK:
        return jsonify({k: JOB[k] for k in ("running", "messages", "result", "error")})


@app.post("/api/reset")
def api_reset():
    with LOCK:
        if JOB["running"]:
            return jsonify({"error": "a trading run is in progress"}), 409
    reset_workspace()
    return jsonify(state_payload())


if __name__ == "__main__":
    ensure_demo_files()
    if "--prefetch" in sys.argv:   # save an EIA copy in advance, so the demo still works if the API is down
        raw, source = fetch_prices()
        if source.startswith("the saved copy"):
            print(f"EIA could not be reached; the existing saved copy (prices up to {raw.index[-1]:%d %b %Y}) was kept.")
            sys.exit(1)
        print(f"Saved EIA data up to {raw.index[-1]:%d %b %Y} from {source} to {CACHE}")
        sys.exit(0)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Live paper-trading demo: {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
