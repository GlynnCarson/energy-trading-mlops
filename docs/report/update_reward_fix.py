"""One-shot script: updates energy_trading_mlops_proposal_v17.docx after the DEF-01 reward fix and the
five-seed retraining (retrain_fixed_reward.py, reports/evaluation_reward_fix/).

Changes the Abstract, Chapter 5 (environment, training, tests, defects, evaluation, Tables 5.5-5.8, new
Table 5.9, objectives table renumbered 5.10), the List of Figures / List of Tables, and Chapter 6. Text is
edited in place, keeping each paragraph's existing run formatting; new paragraphs and tables are copies of
existing ones, so styles match the proposal. Every target is located by its exact opening text and must be
unique, otherwise the script stops without saving.

Backup made before running: docs/report/backup/energy_trading_mlops_proposal_v17_before_reward_fix.docx
Do not rerun: it refuses if Table 5.10 already exists.
"""
import copy
import sys

import docx
from docx.oxml.ns import qn

DOC = "energy_trading_mlops_proposal_v17.docx"
d = docx.Document(DOC)
P = d.paragraphs
body = d.element.body

if any(p.text.startswith("Table 5.10:") for p in P):
    sys.exit("Already updated (Table 5.10 exists); not running again.")

FIRST_CHAPTER = next(i for i, p in enumerate(P) if p.text.startswith("Chapter One:"))


def find(prefix, front=False):
    """Unique paragraph starting with prefix; front=True searches the front matter, else the body."""
    rng = range(0, FIRST_CHAPTER) if front else range(FIRST_CHAPTER, len(P))
    hits = [P[i] for i in rng if P[i].text.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"{len(hits)} matches for: {prefix!r}")
    return hits[0]


def set_text(p, text):
    """Replace a paragraph's text, keeping the first run's formatting."""
    el = p._p if hasattr(p, "_p") else p
    runs = el.findall(qn("w:r"))
    if not runs:
        r = el.makeelement(qn("w:r"), {})
        el.append(r)
        runs = [r]
    for r in runs[1:]:
        el.remove(r)
    r = runs[0]
    for child in list(r):
        if child.tag in (qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:lastRenderedPageBreak")):
            r.remove(child)
    t = r.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)


def insert_after(anchor_el, template_p, text):
    el = copy.deepcopy(template_p._p if hasattr(template_p, "_p") else template_p)
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for b in el.findall(qn(tag)):
            el.remove(b)
    set_text(el, text)
    anchor_el.addnext(el)
    return el


def cell_set(tr, col, text):
    tc = tr.findall(qn("w:tc"))[col]
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    set_text(ps[0], text)


def fill_rows(tbl, rows, start=1):
    """Set rows[k] into table rows start+k, adding copies of the last data row as needed, removing extras."""
    trs = tbl.findall(qn("w:tr"))
    while len(trs) < start + len(rows):
        trs[-1].addnext(copy.deepcopy(trs[-1]))
        trs = tbl.findall(qn("w:tr"))
    for extra in trs[start + len(rows):]:
        tbl.remove(extra)
    trs = tbl.findall(qn("w:tr"))
    for k, vals in enumerate(rows):
        for c, v in enumerate(vals):
            cell_set(trs[start + k], c, v)


def set_widths(tbl, widths):
    for gc, w in zip(tbl.find(qn("w:tblGrid")).findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    for tr in tbl.findall(qn("w:tr")):
        for tc, w in zip(tr.findall(qn("w:tc")), widths):
            tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(w))


def table_after_caption(caption_prefix):
    cap = find(caption_prefix)
    nxt = cap._p.getnext()
    assert nxt.tag == qn("w:tbl"), caption_prefix
    return cap, nxt


def el_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def row_by_id(tbl, ident):
    hits = [tr for tr in tbl.findall(qn("w:tr"))
            if el_text(tr.findall(qn("w:tc"))[0]).strip() == ident]
    assert len(hits) == 1, ident
    return hits[0]


# Collect every target first, so nothing is changed unless all are found ------------------------------
abs1 = find("Electricity prices in liberalised", front=True)
abs2 = find("The system ingests hourly", front=True)
abs3 = find("All cloud infrastructure is provisioned", front=True)
kw = find("Keywords:", front=True)
lof2 = find("Figure 5.2: Backtest Dashboard", front=True)
lof3 = find("Figure 5.3: Validation Return", front=True)
lof4 = find("Figure 5.4: Cumulative Return", front=True)
lot8 = find("Table 5.8: Test-Period Results", front=True)
lot9 = find("Table 5.9: Evaluation of the Specific Objectives", front=True)

p_evidence = find("All results in this chapter come from recorded evidence")
p_step2 = find("ii.  Run pytest tests/ to confirm the environment")
p_step4 = find("iv.  Run python evaluate_eia_runs.py")
p_step5 = find("v.  Run mlflow ui")
p_rerun = find("A rerun of step iii will not reproduce")
p_env = find("EnergyTradingEnv (src/ml/drl/env.py, FR-08)")
p_dag = find("The Airflow DAG (src/pipeline/dag.py")
cap52 = find("Figure 5.2: Backtest dashboard")
p_dash = find("The dashboard shows the cumulative return")
p_runs = find("Five training runs were logged to MLflow")
cap55, t55 = table_after_caption("Table 5.5: Training runs logged in MLflow")
p_fig53 = find("Figure 5.3 shows how the PPO-300k run evolved")
cap53 = find("Figure 5.3: Validation return and entropy loss")
p_after53 = find("The two curves together show that longer training")
p_strategy = find("Testing combined four levels.")
p_suite = find("The automated suite contained 61 unit tests")
cap56, t56 = table_after_caption("Table 5.6: Test cases and results")
cap57, t57 = table_after_caption("Table 5.7: Defect log")
p_def01 = find("DEF-01 is the most important finding of the project.")
p_eval_intro = find("The two agents trained on real data were evaluated")
cap58, t58 = table_after_caption("Table 5.8: Test-period results")
cap54 = find("Figure 5.4: Cumulative return over the full test period")
p_onpaper = find("On paper, PPO-300k performed very well")
p_three = find("Three further pieces of evidence show")
p_10k = find("PPO-10k, trained for only 10,000 steps")
p_repo = find("The repository contains the source code, tests")
p_t59ref = find("Table 5.9 evaluates each specific objective")
cap59, t59 = table_after_caption("Table 5.9: Evaluation of the specific objectives")
p_dataops = find("The data and MLOps parts of the project produced the strongest evidence.")
p_rq3 = find("The trading results do not answer research question iii")
p_lim2 = find("ii.  Each configuration was trained once with a single seed")
p_lim6 = find("vi.  Because of DEF-01, all trading results describe")

c6_obj3 = find("Objective iii was partially achieved, and the answer to research question iii is no.")
c6_obj4 = find("Objective iv was partially achieved. Experiment tracking")
c6_obj5 = find("Objective v was partially achieved. All 61 unit tests")
c6_contrib = find("The project makes two contributions.")
c6_bounds = find("These conclusions hold only within the conditions that were tested")
c6_r1 = find("i.  Correct the reward before any further training")
c6_r2 = find("ii.  Correct the evaluation defects before reporting new results")
c6_r3 = find("iii.  Do not use the agent's outputs for trading decisions")
c6_r5 = find("v.  Execute the orchestration and deployment code")
c6_r6 = find("vi.  Base the planned quality gate")
c6_r7 = find("vii.  Revise data preparation for spike-driven commodities")
c6_r8 = find("viii.  Report DRL trading results against a distribution")
c6_r10 = find("x.  No hardware change is recommended")
c6_fw_intro = find("Future work should extend the study once the essential recommendations")
c6_fw1 = find("i.  A controlled comparison of reward designs.")
c6_fw2 = find("ii.  Robustness across seeds and time periods.")
c6_summary = find("In summary, the project delivered a reproducible data and experiment pipeline")

# Abstract --------------------------------------------------------------------------------------------
set_text(abs1, "Energy commodity prices are highly volatile, yet most machine learning models for energy trading are "
         "one-off scripts that cannot be scheduled, retrained or reproduced. This project designed and developed "
         "the Energy Trading MLOps Platform, an end-to-end Machine Learning Operations (MLOps) pipeline that trains "
         "and evaluates a Deep Reinforcement Learning (DRL) agent for trading natural gas.")
set_text(abs2, "The pipeline ingests daily Henry Hub natural gas and West Texas Intermediate crude oil prices for 2010 "
         "to 2026 from the U.S. Energy Information Administration (EIA) Application Programming Interface (API), "
         "validates them and engineers 38 leakage-free features. A Proximal Policy Optimisation (PPO) agent is "
         "trained in a custom Gymnasium trading environment with transaction costs, every run is tracked in MLflow, "
         "and the agent is backtested against Buy & Hold and random baselines. Apache Airflow orchestration, Docker "
         "containers, Terraform infrastructure and GitHub Actions workflows were implemented as code but were not "
         "used for the reported results.")
set_text(abs3, "Testing revealed that the original reward contained only transaction costs, so the first agent never "
         "learned from prices and its +124 % test return came from a single price spike. After the reward was "
         "corrected, five agents trained with different random seeds all beat Buy & Hold on the held-out test "
         "period from December 2024 to July 2026, with a mean return of +135 % against +8.7 % and a higher Sharpe "
         "ratio in every run, although returns varied widely between seeds. The results hold for one market and "
         "one simulated test period. The project shows that reproducible experiment tracking is what exposed the "
         "defect and made the corrected result credible.")
set_text(kw, "Keywords: Deep Reinforcement Learning, PPO, MLOps, Energy Trading, Henry Hub Natural Gas, MLflow, "
         "Reproducibility, Backtesting.")

# Lists of figures and tables ---------------------------------------------------------------------------
set_text(lof2, "Figure 5.2: Backtest Dashboard for the Fixed-Reward PPO-300k Agent, Seed 42 (First 252 Test Days)")
set_text(lof3, "Figure 5.3: Validation Return and Entropy Loss during PPO-300k Training, before and after the "
         "Reward Fix")
set_text(lof4, "Figure 5.4: Cumulative Return over the Full Test Period before and after the Reward Fix, with "
         "Price and Position")
set_text(lot8, "Table 5.8: Test-Period Results before and after the Reward Fix")
insert_after(lot8._p, lot8, "Table 5.9: Results of the Five Fixed-Reward PPO-300k Runs")
set_text(lot9, "Table 5.10: Evaluation of the Specific Objectives")

# 5.1 and 5.2.3 -----------------------------------------------------------------------------------------
set_text(p_evidence, "All results in this chapter come from recorded evidence: the MLflow tracking store (./mlruns), "
         "the pytest output and two evaluation scripts. evaluate_eia_runs.py rebuilt the dataset and re-ran the "
         "first saved models, reproducing the logged MLflow metrics exactly (Section 5.6.4). retrain_fixed_reward.py "
         "retrained the agents on the same dataset after the reward defect DEF-01 was fixed, and evaluated them "
         "with the same protocol. These records are the basis for trusting the numbers reported here.")
set_text(p_step2, "ii.  Run pytest tests/ to confirm the environment (65 tests are expected to pass).")
set_text(p_step5, "v.  Run python retrain_fixed_reward.py train --budget 300000 --seed 42, and again with seeds 1 "
         "to 4, to retrain the agents with the corrected reward on the same dataset, then python "
         "retrain_fixed_reward.py summary to pool the results in reports/evaluation_reward_fix/.")
insert_after(p_step5._p, p_step5, "vi.  Run mlflow ui --backend-store-uri ./mlruns to inspect the runs.")
set_text(p_rerun, "A rerun of step iii will not reproduce the reported numbers exactly unless the data is cut at the "
         "same date, because the EIA series grows every day; evaluate_eia_runs.py handles this by trimming the "
         "data until the split sizes match those logged in MLflow, and retrain_fixed_reward.py reuses the same "
         "trimmed dataset. The Docker Compose stack can be started with docker compose up, and its configuration "
         "was validated with docker compose config.")

# 5.4.3, 5.4.5, 5.4.6 --------------------------------------------------------------------------------------
set_text(p_env, "EnergyTradingEnv (src/ml/drl/env.py, FR-08) implements the Gymnasium interface. On each step it "
         "converts the action into a target position, clamps the trade to the available cash or the position held, "
         "ignores trades smaller than the minimum trade size, charges a transaction cost of 0.1 % of the trade value "
         "and updates cash, position and average entry price. The day then advances, the portfolio is valued at the "
         "new price, and the reward is the change in portfolio value multiplied by a scaling factor of 0.0001, so it "
         "contains both the profit or loss of the position and the transaction cost; a penalty of 1 is subtracted if "
         "the stop-loss is triggered. In the first implementation the portfolio was valued before the day advanced, "
         "so the reward contained only the costs (DEF-01, Section 5.6.3). The observation is built from the feature "
         "vector, normalised with a running mean and variance (RunningNorm) and clipped to five standard deviations, "
         "followed by the cash ratio, position ratio and normalised unrealised profit. In training mode each episode "
         "starts at a random point; in test mode it starts at the beginning of the data. The environment passes the "
         "Gymnasium check_env validation (test test_gymnasium_check_env).")
txt = p_dag.text
old = ("and the DAG imports DummyOperator, which is deprecated in Airflow 2.8 (DEF-04).")
assert old in txt, "DAG sentence"
set_text(p_dag, txt.replace(old, "and the DAG imported DummyOperator, which is deprecated in Airflow 2.8 "
                                 "(DEF-04, since fixed)."))
set_text(cap52, "Figure 5.2: Backtest dashboard produced by backtest.py for the fixed-reward PPO-300k agent, seed 42 "
         "(first 252 test days)")
set_text(p_dash, "The dashboard shows the cumulative return of the agent and the baselines, the price with buy and "
         "sell markers, the position held, the drawdown and the distribution of actions. Its axis labels are taken "
         "from the run configuration, so this daily run is shown in trading days and $/MMBtu; the report produced in "
         "July labelled the axes in hours and USD/MWh, which was corrected during evaluation (DEF-05). Two further "
         "plotting faults were corrected when the figure was regenerated for the retrained agent: every panel was "
         "cut to the length of the shortest series, so a Random Agent that hit the stop-loss after 43 days hid the "
         "rest of the agent's curve, and the trade markers were drawn one day late (DEF-10). The figure is "
         "interpreted in Section 5.6.4.")

# 5.5 -----------------------------------------------------------------------------------------------------
set_text(p_runs, "Eleven training runs were logged to MLflow (Table 5.5). The first five were made in July 2026: two "
         "smoke tests, the 500,000-step synthetic run and the two runs on EIA data, PPO-10k and PPO-300k, all trained "
         "with the original cost-only reward. The other six were made in November 2026 after the reward was "
         "corrected (DEF-01): PPO-10k with seed 42, and PPO-300k with seed 42, the seed used in July, and with seeds "
         "1 to 4, to measure how much the result depends on the random seed. Seeds 1 to 4 were trained in parallel, "
         "which explains their longer durations. Every run used the same data split and the hyperparameters in "
         "config/drl.yaml (learning rate 3e-4, 2,048 steps per rollout, batch size 64, 10 epochs, discount factor "
         "0.99, clip range 0.2, entropy coefficient 0.01 and a [256, 256, 128] network), with the default "
         "Stable-Baselines3 PPO settings for anything not listed. No hyperparameter search was performed. The "
         "checkpoint used for testing was the best checkpoint chosen by EvalCallback, which selects the checkpoint "
         "with the highest mean reward on the validation split; the test split played no part in selection. The "
         "validation return in Table 5.5 is that of each run's final model, which was not selected on the "
         "validation data.")
cell_set(row_by_id(t55, "PPO-10k"), 0, "PPO-10k, cost-only reward")
cell_set(row_by_id(t55, "PPO-300k"), 0, "PPO-300k, cost-only reward")

new_rows = [
    ("PPO-10k, fixed reward, seed 42", "EIA daily gas", "10,000 (4)", "1 min 4 s", "+0.75 %"),
    ("PPO-300k, fixed reward, seed 42", "EIA daily gas", "300,000 (4)", "19 min 38 s", "+43.78 %"),
    ("PPO-300k, fixed reward, seed 1", "EIA daily gas", "300,000 (4)", "29 min 38 s", "+44.30 %"),
    ("PPO-300k, fixed reward, seed 2", "EIA daily gas", "300,000 (4)", "29 min 19 s", "+15.31 %"),
    ("PPO-300k, fixed reward, seed 3", "EIA daily gas", "300,000 (4)", "29 min 19 s", "+54.88 %"),
    ("PPO-300k, fixed reward, seed 4", "EIA daily gas", "300,000 (4)", "29 min 20 s", "+15.64 %"),
]
for vals in new_rows:
    last = t55.findall(qn("w:tr"))[-1]
    nr = copy.deepcopy(last)
    last.addnext(nr)
    for c, v in enumerate(vals):
        cell_set(nr, c, v)

set_text(p_fig53, "Figure 5.3 compares the July PPO-300k run with the five fixed-reward PPO-300k runs. Under the "
         "cost-only reward, the return on the validation episode rose to +16.7 % at 60,000 steps and then fell, "
         "ending at -6.9 % at 300,000 steps, while the entropy loss rose steadily towards zero, which means the "
         "policy became more and more deterministic. With the corrected reward, every run stayed above the Buy & "
         "Hold return of -8.0 % on the validation period for almost the whole of training, and the final models "
         "returned between +15.31 % and +54.88 %. Their entropy loss also rose, but more slowly.")
set_text(cap53, "Figure 5.3: Validation return and entropy loss during PPO-300k training, July cost-only run and "
         "five fixed-reward runs")
set_text(p_after53, "Under the cost-only reward, longer training did not produce a better trader: the policy "
         "converged towards holding nothing, the cheapest behaviour, because the reward did not contain the profit "
         "or loss from price movements (Section 5.6.3). Once the reward included it, training improved the "
         "validation result in every run, although the spread between seeds was wide.")

# 5.6.1-5.6.3 ---------------------------------------------------------------------------------------------
txt = p_strategy.text
old = "the reward test (TC-11) was a white-box test designed after inspecting the step function."
assert old in txt
set_text(p_strategy, txt.replace(old, "the reward test (TC-11) was a white-box test designed after inspecting the "
                                      "step function, and was later automated as three regression tests."))
set_text(p_suite, "The automated suite contains 65 unit tests: 8 for ingestion, 26 for transformation and 31 for the "
         "environment, training and backtest code, including the four regression tests added with the DEF-01 and "
         "DEF-09 fixes. All 65 passed in 58 seconds, with 62 % statement coverage of src/. Coverage was highest for "
         "the environment (93 %), training (86 %) and transformation (83 %) modules, and zero for the DAG and the "
         "Airflow task wrappers, which need Airflow to run. Table 5.6 groups the unit tests into test cases and adds "
         "the integration, system and configuration tests.")
r = row_by_id(t56, "TC-07")
cell_set(r, 2, "5 tests: metrics keys, positive return sign, Buy & Hold and Random baselines run, and a seeded "
         "Random Agent gives the same result twice.")
cell_set(r, 3, "5/5 passed. Pass")
cell_set(row_by_id(t56, "TC-09"), 3, "-2.95 % and -15.04 %, identical to the logged values; the six fixed-reward "
         "runs also matched their logged values. Pass")
cell_set(row_by_id(t56, "TC-10"), 3, "14 min 57 s; after the reward fix 19 min 38 s, or about 29 min 30 s with four "
         "runs in parallel. Pass")
cell_set(row_by_id(t56, "TC-11"), 3, "Before the fix: portfolio rose from 100,000 to 107,000 but every reward was 0. "
         "Fail (DEF-01). Retest after the fix: 3 automated tests (rising price, falling price, rewards sum to the "
         "profit) pass. Pass")
cell_set(row_by_id(t57, "DEF-01"), 3, "Fixed: the reward is computed after moving to the next price; three regression "
         "tests added; all EIA agents retrained (Section 5.6.4).")
cell_set(row_by_id(t57, "DEF-04"), 3, "Fixed in code: the DAG imports EmptyOperator. Not yet retested under Airflow.")
cell_set(row_by_id(t57, "DEF-09"), 3, "Fixed: the action sampler is seeded; results are reproducible.")
for ident, col, text in [
    ("DEF-01", 2, "In step(), the previous and new portfolio values were both calculated at the same day's price, and "
                  "the price change happened only after the step ended. The reward therefore contained only "
                  "transaction costs and the stop-loss penalty, never the profit or loss from price movements, so the "
                  "agents were trained to avoid trading costs rather than to trade profitably."),
    ("DEF-04", 2, "The DAG imported the deprecated DummyOperator instead of EmptyOperator."),
    ("DEF-09", 2, "The Random Agent seeded the environment but not the action sampler, so its results differed "
                  "between executions."),
    ("DEF-02", 3, "Mitigated in evaluate_eia_runs.py and retrain_fixed_reward.py; source fix pending."),
]:
    cell_set(row_by_id(t57, ident), col, text)
r9 = row_by_id(t57, "DEF-09")
r10 = copy.deepcopy(r9)
r9.addnext(r10)
for c, v in enumerate(["DEF-10", "Low", "The backtest dashboard cut every panel to the length of the shortest series, "
                       "so a Random Agent stopped out early hid most of the agent's curve, and it drew trade markers "
                       "one day late.", "Fixed: each series is plotted over its own length; Figure 5.2 regenerated."]):
    cell_set(r10, c, v)
set_text(p_def01, "DEF-01 is the most important finding of the project. It was discovered while trying to explain why "
         "the July PPO-300k agent stopped trading, and test TC-11 confirmed it: with zero fees, a full position held "
         "while the price rose eightfold produced a reward of exactly zero on every step. Every agent trained before "
         "the fix therefore received no signal that linked positions to price movements. The fix values the "
         "portfolio after moving to the next price, three regression tests now guard it, and the EIA agents were "
         "retrained on the same data. The results before and after the fix are compared in Section 5.6.4.")

# 5.6.4 ---------------------------------------------------------------------------------------------------
set_text(p_eval_intro, "The agents trained on real data were evaluated on the held-out test period with their best "
         "checkpoints. Because of DEF-02 the original backtest covered only the first 252 test days, so every agent "
         "was evaluated on that window and on the full test period of 414 trading steps (4 December 2024 to 6 July "
         "2026), against Buy & Hold and 30 seeded Random Agents. Table 5.8 compares the July agent trained with the "
         "cost-only reward with the agents retrained after the fix, Table 5.9 gives the result of each of the five "
         "fixed-reward PPO-300k runs, and Figure 5.4 shows the full-period results. Buy & Hold differs slightly from "
         "the July evaluation (+8.68 % instead of +9.93 % over the full period) because the fix values the "
         "portfolio one day later.")
set_text(cap58, "Table 5.8: Test-period results before and after the reward fix")
set_widths(t58, [1500, 1884, 1656, 1152, 1800, 1080])
W1, WF = "First 252 days", "Full test period"
fill_rows(t58, [
    (W1, "PPO-300k, cost-only (July)", "+126.37 %", "1.24", "-6.43 %", "38"),
    (W1, "PPO-300k, fixed, seed 42", "+27.77 %", "0.81", "-53.44 %", "222"),
    (W1, "PPO-300k, fixed, mean of 5 seeds", "+109.25 %", "1.42", "-27.82 %", "232"),
    (W1, "PPO-10k, fixed, seed 42", "+27.15 %", "0.69", "-43.33 %", "252"),
    (W1, "Buy & Hold", "+25.68 %", "0.78", "-66.04 %", "1"),
    (WF, "PPO-300k, cost-only (July)", "+124.02 %", "0.96", "-6.43 %", "42"),
    (WF, "PPO-300k, fixed, seed 42", "+124.36 %", "1.15", "-53.44 %", "370"),
    (WF, "PPO-300k, fixed, mean of 5 seeds", "+135.00 %", "1.16", "-36.07 %", "375"),
    (WF, "PPO-10k, fixed, seed 42", "+23.37 %", "0.54", "-43.38 %", "414"),
    (WF, "Buy & Hold", "+8.68 %", "0.66", "-68.19 %", "1"),
])
set_text(cap54, "Figure 5.4: Cumulative return over the full test period of the five fixed-reward agents, the July "
         "cost-only agent and Buy & Hold (top), with the Henry Hub price (middle) and the seed-42 position (bottom)")

set_text(p_onpaper, "Before the fix, PPO-300k appeared to perform very well: +124.02 % over the full test period "
         "against +9.93 % for Buy & Hold in the July evaluation, with a maximum drawdown of only -6.43 %. Almost all "
         "of this came from one price movement. The agent held a large position during the first weeks of the test "
         "period, was still holding it when the price jumped from $4.30 per MMBtu on 16 January 2025 to $9.86 on 17 "
         "January, sold at the high price and then stayed out of the market, holding nothing on 377 of the 414 test "
         "days. The same agent lost 15.04 % on the validation period, and because of DEF-01 it had never been "
         "rewarded for profit, so the result could not be attributed to a learned trading skill: the policy had "
         "converged towards holding nothing, the cheapest behaviour under a cost-only reward.")
set_text(p_three, "After the fix the agents behaved differently. The fixed-reward PPO-300k agents traded on almost "
         "every day and held no position on between 34 and 74 of the 414 test days. All five beat Buy & Hold over "
         "the full test period, with a mean return of +135.00 % (standard deviation 48.25 points) against +8.68 %, "
         "a higher Sharpe ratio in every run (0.97 to 1.38 against 0.66) and a smaller maximum drawdown (-25.35 % to "
         "-53.44 % against -68.19 %). All five were also profitable on the validation period, where Buy & Hold lost "
         "8.05 %. The 30 seeded Random Agents returned between -58.74 % and +151.16 % over the full period (mean "
         "+4.61 %); no more than 7 of them beat any of the five runs, and none beat seeds 1 and 3. Table 5.9 gives "
         "the result of each run.")

# new Table 5.9 (copy of Table 5.8's caption and table), placed after the paragraph above
cap9 = copy.deepcopy(cap58._p)
set_text(cap9, "Table 5.9: Results of the five fixed-reward PPO-300k runs")
p_three._p.addnext(cap9)
t9 = copy.deepcopy(t58)
cap9.addnext(t9)
set_widths(t9, [1100, 1650, 1650, 1650, 1172, 1850])
header = t9.findall(qn("w:tr"))[0]
for c, v in enumerate(["Seed", "Validation return", "Test, first 252 days", "Test, full period", "Sharpe (full)",
                       "Max drawdown (full)"]):
    cell_set(header, c, v)
fill_rows(t9, [
    ("42", "+43.78 %", "+27.77 %", "+124.36 %", "1.15", "-53.44 %"),
    ("1", "+44.30 %", "+205.67 %", "+170.79 %", "1.38", "-25.35 %"),
    ("2", "+15.31 %", "+60.17 %", "+121.93 %", "1.31", "-25.41 %"),
    ("3", "+54.88 %", "+141.13 %", "+190.91 %", "1.02", "-39.20 %"),
    ("4", "+15.64 %", "+111.52 %", "+67.02 %", "0.97", "-36.97 %"),
    ("Mean ± SD", "+34.78 ± 18.17 %", "+109.25 ± 69.58 %", "+135.00 ± 48.25 %", "1.16 ± 0.18", "-36.07 ± 11.63 %"),
    ("Buy & Hold", "-8.05 %", "+25.68 %", "+8.68 %", "0.66", "-68.19 %"),
])
p_var = insert_after(t9, p_three, "The results are not uniform. Full-period returns ranged from +67.02 % to "
         "+190.91 %, and over the first 252 days seed 42 only matched Buy & Hold (+27.77 % against +25.68 %): it "
         "added to its position at the $9.86 peak of the January 2025 spike and lost about $58,000 when the price "
         "fell back to $4.40 on 21 January (Figure 5.2). Most of the variation between seeds arises in the two spike "
         "periods: over January 2025 and January to February 2026 the five agents' results ranged from a loss of "
         "about $38,000 to a gain of about $101,000. Outside those months all five gained more than Buy & Hold, "
         "between about $55,000 and $118,000 against $46,000, so the advantage did not come from the spikes alone. "
         "Figure 5.4 shows these paths: the fixed-reward agents pull away from Buy & Hold gradually through 2025, "
         "while the July agent's curve is flat after its single profitable exit.")
set_text(p_10k, "PPO-10k, retrained for only 10,000 steps, kept an almost constant position of about half the "
         "maximum and traded every day; its +23.37 % full-period return came almost entirely from the price spikes, "
         "and 11 of the 30 Random Agents beat it. This is the behaviour of an almost untrained policy whose actions "
         "stay near zero, which maps to a half-sized position. On the synthetic hourly data, the 500,000-step agent "
         "trained in July with the cost-only reward returned +0.11 % on its 720-hour test window against +0.03 % for "
         "Buy & Hold, with a smaller drawdown (-1.67 % against -3.21 %); it was not retrained.")

# 5.7, 5.8 ------------------------------------------------------------------------------------------------
txt = p_repo.text
old = "the report-generation scripts in docs/report/"
assert old in txt
set_text(p_repo, txt.replace(old, "the evaluation scripts evaluate_eia_runs.py and retrain_fixed_reward.py, the "
                                  "report-generation scripts in docs/report/"))
set_text(p_t59ref, "Table 5.10 evaluates each specific objective against its evidence.")
set_text(cap59, "Table 5.10: Evaluation of the specific objectives")
r3 = [tr for tr in t59.findall(qn("w:tr")) if el_text(tr).startswith("iii.")][0]
cell_set(r3, 1, "Environment and PPO training implemented and tested (TC-05, TC-06); the reward first excluded "
         "profit and loss (DEF-01) and was corrected (TC-11); five retrained agents beat Buy & Hold on validation "
         "and test data; idle and drawdown penalties not implemented.")
r5 = [tr for tr in t59.findall(qn("w:tr")) if el_text(tr).startswith("v.")][0]
cell_set(r5, 1, "65 unit tests passed; held-out backtests reproduced; integration and system tests run; cloud and "
         "CI/CD defined but not executed (TC-16).")
set_text(p_dataops, "The data and MLOps parts of the project produced the strongest evidence. The pipeline retrieved "
         "and prepared 16 years of real market data without leakage, and the experiment tracking was good enough to "
         "reconstruct the exact dataset of a run made four months earlier and reproduce its logged results to four "
         "decimal places. This supports the argument in Chapter 2, drawn from Kreuzberger et al. (2023) and Sculley "
         "et al. (2015), that tracking and reproducibility are what separate a maintainable ML system from a one-off "
         "script. In this project it was the tracking evidence that exposed the reward defect: without the logged "
         "configuration and models, the +124 % backtest result could easily have been reported as a success. The "
         "same records then allowed the agents to be retrained on exactly the same data after the fix, so the "
         "before-and-after comparison changes only the reward.")
set_text(p_rq3, "Research question iii can now be answered for this market and period. Because the corrected reward "
         "is the scaled change in portfolio value, the cumulative reward over an episode equals the scaled profit or "
         "loss, so every fixed-reward PPO-300k agent achieved a positive cumulative reward on the unseen test data, "
         "whereas no agent trained before the fix could. The evidence that the agents learned a useful behaviour is "
         "much stronger than before: all five seeds beat Buy & Hold on both the validation and the test period, and "
         "outside the spike months, and few Random Agents beat them. It is limited by the wide spread between seeds "
         "and by the weight of a few days in each result. The results are consistent with Cao, Chen and Fu (2020), "
         "who found that DRL agents trained with transaction costs in the reward produced realistic trading "
         "behaviour, but one market and one test period are not enough to confirm that finding.")
set_text(p_lim2, "ii.  The fixed-reward PPO-300k configuration was trained with five seeds, which gave a wide spread "
         "of results (+67.02 % to +190.91 % over the full test period); five seeds are too few to measure that "
         "spread precisely, the other configurations were trained once, and no hyperparameter search was performed.")
set_text(p_lim6, "vi.  The idle and drawdown penalties proposed in Chapter 3 were not implemented, so the reward does "
         "not penalise risk directly; the fixed-reward agents had maximum drawdowns of up to -53.44 %.")

# Chapter 6 -----------------------------------------------------------------------------------------------
set_text(c6_obj3, "Objective iii was partially achieved, and within the conditions tested the answer to research "
         "question iii is yes. The trading environment and PPO training were implemented and tested (TC-05 and "
         "TC-06), but the idle and drawdown penalties of the proposed reward were not implemented. Testing also "
         "found that the first implementation of the reward never included the profit or loss from price movements "
         "(DEF-01), so the agents trained in July learned only to avoid costs, and their apparently strong +124.02 % "
         "test return came from a single price spike. After the reward was corrected, five PPO-300k agents trained "
         "with different seeds all achieved a positive cumulative reward on the unseen test period and all beat Buy "
         "& Hold on both the validation and the test period: their mean test return was +135.00 % against +8.68 %, "
         "with a higher Sharpe ratio and a smaller maximum drawdown in every run, and no more than 7 of 30 random "
         "agents beat any of them. The supportable conclusion is that PPO with the corrected reward learned a "
         "trading behaviour that outperformed Buy & Hold on Henry Hub gas over this test period, but the wide spread "
         "between seeds (+67.02 % to +190.91 %) and the weight of a few spike days mean that the size of the "
         "advantage is uncertain.")
txt = c6_obj4.text
old = "without them, the +124.02 % backtest could easily have been reported as a success."
assert old in txt
set_text(c6_obj4, txt.replace(old, "without them, the +124.02 % backtest could easily have been reported as a "
                                   "success, and with them the agents could be retrained on exactly the same data "
                                   "once the defect was fixed."))
set_text(c6_obj5, "Objective v was partially achieved. All 65 unit tests pass with 62 % statement coverage of the "
         "source code, the live EIA client and the full data-to-backtest path were tested, the Docker Compose file "
         "was validated (TC-15) and the held-out results were reproduced. Research question v is answered only in "
         "part. The most important lesson from testing is that a passing unit suite did not show the system was "
         "correct: the critical reward defect was found by investigating why the agent stopped trading and was then "
         "confirmed by a white-box test of the reward (TC-11), after all 61 unit tests of the time had passed; that "
         "test is now part of the suite. Ten defects were recorded in total, of which four have been fixed. The DAG, "
         "the Terraform configuration and the CI/CD workflows were not executed for the reported results, so the "
         "orchestration, cloud and deployment requirements rest on configuration evidence only.")
set_text(c6_contrib, "The project makes two contributions. The first is a tested and reproducible data-to-backtest "
         "pipeline for DRL trading on daily EIA energy prices, in which every reported result can be traced to a "
         "logged run and rebuilt from it. The second is the evaluation itself. It documents how a defect in the "
         "reward produced a backtest that looked far better than both baselines, it shows which checks exposed the "
         "problem (reproducible tracking, a distribution of random agents rather than a single one, validation "
         "results read alongside test results, and a white-box test of the reward), and it shows that once the "
         "defect was fixed and the agents were retrained on the same data, a comparison across five seeds gave a "
         "result that held in every run but varied widely between them. This supports the position taken in "
         "Chapter 2 that tracking and reproducibility are as important to a working ML system as the model.")
txt = c6_bounds.text
old = "PPO trained once per configuration with seed 42 and no hyperparameter search;"
assert old in txt
set_text(c6_bounds, txt.replace(old, "the fixed-reward PPO-300k configuration trained with five seeds and no "
                                     "hyperparameter search;"))
set_text(c6_r1, "i.  Keep the reward regression tests and retrain whenever the environment changes (system "
         "developers; highest priority). The reward defect DEF-01 has been fixed and is guarded by three automated "
         "tests (TC-11). Any later change to the step function, the reward or the data should be followed by the "
         "full test suite and by retraining and re-evaluating the agents on the same data with the same protocol, "
         "because Chapter 5 shows that a reward error can produce results that look credible.")
set_text(c6_r2, "ii.  Correct the remaining evaluation defects before reporting new results (system developers). The "
         "backtest should run over the whole test split rather than stopping at the episode length (DEF-02), and the "
         "drawdown logged during training should be calculated on portfolio values (DEF-03). DEF-02 caused the "
         "original backtest to cover only 252 of the 414 test days and is currently worked around in the evaluation "
         "scripts, and DEF-03 logs meaningless values such as -64,450,496.7 %. The Random Agent's seeding (DEF-09) "
         "has been fixed.")
set_text(c6_r3, "iii.  Do not use the agent's outputs for trading decisions (prospective users such as energy traders "
         "and analysts). The agent should be treated as a research prototype that has been tested in simulation "
         "only. The fixed-reward agents beat Buy & Hold on one market and one test period, but with maximum "
         "drawdowns of up to -53.44 % and results that varied widely between seeds. Any later version should keep a "
         "human trader responsible for every decision until it has been evaluated as described in Section 6.4.")
txt = c6_r5.text
old = "The DAG should import EmptyOperator instead of the deprecated DummyOperator (DEF-04) and then be run under Airflow,"
assert old in txt
set_text(c6_r5, txt.replace(old, "The DAG now imports EmptyOperator instead of the deprecated DummyOperator (DEF-04) "
                                 "and should be run end to end under Airflow,"))
txt = c6_r6.text
old = "could never be passed under the implemented reward, while a gate based on test return alone would have promoted PPO-300k"
assert old in txt
set_text(c6_r6, txt.replace(old, "could never be passed under the original cost-only reward, while a gate based on "
                                 "test return alone would have promoted the July PPO-300k agent"))
txt = c6_r7.text
old = "including the true January 2026 peak."
assert old in txt
set_text(c6_r7, txt.replace(old, "including the true January 2026 peak, a period in which the fixed-reward agents' "
                                 "results differed most."))
set_text(c6_r8, "viii.  Report DRL trading results across seeds and against a distribution, not a single run "
         "(researchers and students reusing this work). On this test period the returns of 30 random agents ranged "
         "from -58.74 % to +151.16 %, and two seeds of the same configuration differed by more than 100 percentage "
         "points, so a single backtest return says little on its own. Results should be reported over several "
         "training seeds, on both the validation and test periods, against Buy & Hold and a seeded set of random "
         "agents, together with the share of return that came from the largest price moves.")
txt = c6_r10.text
old = "The 300,000-step training run took 14 minutes 57 seconds on a CPU,"
assert old in txt
set_text(c6_r10, txt.replace(old, "A 300,000-step training run took 15 to 20 minutes on a CPU, or about 30 minutes "
                                  "with four runs in parallel,"))
set_text(c6_fw_intro, "Future work should extend the study; correcting the remaining evaluation defects in Section "
         "6.3.1 is repair rather than future work. The directions below are listed in order of priority, and each "
         "arises from a limitation or open question in Section 5.8.")
set_text(c6_fw1, "i.  A controlled comparison of reward designs. Research question iii has been answered only for a "
         "reward that measures profit and loss after costs. PPO should now be compared under other reward designs, "
         "for example with the idle and drawdown penalties proposed in Chapter 3 or a risk-adjusted reward, using "
         "the same data, splits, hyperparameters and seeds so that only the reward changes. Each design should be "
         "judged on validation and test return, maximum drawdown and the share of random agents it beats. Because "
         "the current agents reach drawdowns of up to -53.44 %, this would show whether risk can be reduced without "
         "losing the advantage over Buy & Hold.")
set_text(c6_fw2, "ii.  Robustness across seeds and time periods. The 414-day test period was dominated by two price "
         "spikes, and the five fixed-reward seeds gave full-period returns from +67.02 % to +190.91 %. A "
         "walk-forward evaluation, in which the agent is retrained and tested on successive periods of the 2010 to "
         "2026 data with ten or more seeds per period, would show whether the advantage holds outside this period "
         "and how large it really is. It needs no new data, only more training time, and success would be a return "
         "above Buy & Hold in most periods with a spread across seeds that is small compared with that margin.")
set_text(c6_summary, "In summary, the project delivered a reproducible data and experiment pipeline whose own records "
         "revealed that its first trading agent had not been trained to trade, and which then made it possible to "
         "correct the defect and show, across five seeds, that the retrained agent outperformed Buy & Hold over the "
         "test period. The most valuable next step is therefore not a larger model or more infrastructure, but "
         "testing whether that advantage holds across other periods, markets and reward designs.")

d.save(DOC)
print("Report updated.")
