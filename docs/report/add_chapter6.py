"""One-shot script: inserts Chapter Six (Conclusions, Recommendations and Future
Works) into energy_trading_mlops_proposal_v17.docx, between Section 5.8 and
References.

Formatting is copied from the existing Chapter Five paragraphs (page-break
paragraph, Heading 1/2/3 and justified 1.5-spaced body text), so the new
chapter matches the proposal's styles exactly. Content follows the course
guide "Research Writing for CS 2 Projects - 2026", Chapter 6: no new results,
experiments or citations; every statement traces to Chapter 5 evidence.

Do not rerun: it refuses to run if Chapter Six already exists.
"""
import copy
import sys

import docx
from docx.oxml.ns import qn

DOC = "energy_trading_mlops_proposal_v17.docx"

H1 = "Chapter Six: Conclusions, Recommendations and Future Works"

# (kind, text). kind: h2 / h3 = (number, title); p = body paragraph.
CONTENT = [
    ("h2", ("6.1", "Introduction")),
    ("p", "This chapter closes the report by linking the problem and objectives set out in Chapter 1 to the system described in Chapters 4 and 5 and to the evidence reported there. Section 6.2 states what the project established, answering each specific objective and research question in the order used throughout the report. Section 6.3 sets out recommendations for the people who would maintain, use or extend the system, and Section 6.4 proposes future work. No new experiments, results or sources are introduced: every statement in this chapter rests on evidence or a limitation already reported in Chapter 5."),

    ("h2", ("6.2", "Conclusions")),
    ("p", "The project set out to address three problems identified in Chapter 1: energy price models are usually one-off scripts that cannot be scheduled, retrained or reproduced; forecasting models optimise prediction accuracy rather than trading objectives; and no reference implementation combined live energy data, DRL training, experiment tracking and deployment automation in one system. The aim was an end-to-end MLOps pipeline covering data ingestion, model training, experiment tracking, automated deployment and cloud provisioning. The completed system is a local prototype. It ingests daily Henry Hub natural gas and WTI crude oil prices from the EIA API, prepares a leakage-free feature set, trains a PPO agent in a custom Gymnasium trading environment, tracks every run in MLflow and backtests the agent against Buy & Hold and Random baselines. The orchestration, container, cloud and CI/CD components were written as code but were not executed."),

    ("h3", ("6.2.1", "Findings for Each Objective")),
    ("p", "Objective i and research question i were achieved. The literature review in Chapter 2 found that machine learning models outperform classical statistical models on volatile energy prices, that DRL frames trading as a sequential decision problem in which costs and risk can be built into the reward, and that the main obstacle to using either in practice is operational, because models are rarely versioned, scheduled or monitored. It identified the gap this project targeted: the lack of a single reproducible system that joins energy data ingestion, DRL training and MLOps tooling."),
    ("p", "Objective ii was achieved, with a change of data frequency. The pipeline retrieved 4,166 business-day records from 4 January 2010 to 6 July 2026 from the live EIA API and produced a 4,138-row dataset with 38 features, split chronologically into training, validation and test periods. Research question ii is answered by the design of the features and splits: every lag, rolling statistic and indicator is computed on the previous day's price and the three periods do not overlap in time, which the 34 data tests (TC-01 to TC-04) confirmed. Daily gas prices replaced the hourly electricity data of the proposal because the proposed API route returned only monthly data without a price target. Two weaknesses remain: the six-standard-deviation outlier rule replaced five genuine price spikes, and four calendar features are constant on daily data (DEF-06)."),
    ("p", "Objective iii was partially achieved, and the answer to research question iii is no. The trading environment and PPO training were implemented and tested (TC-05 and TC-06), and five PPO runs were completed and logged. However, the idle and drawdown penalties of the proposed reward were not implemented, and the implemented reward never included the profit or loss from price movements (DEF-01); it contained only transaction costs and the stop-loss penalty. Under that reward the cumulative reward cannot be positive. Measured by portfolio return instead, the best agent, PPO-300k, returned +124.02 % over the 414-day test period against +9.93 % for Buy & Hold, with a maximum drawdown of -6.43 % against -68.19 %. This result does not show a learned trading skill. Almost all of it came from one price spike in January 2025, the agent held no position on 377 of the 414 test days, the same agent lost 15.04 % on the validation period, and 10 % of 30 random agents did better on the same test period. The supportable conclusion is that the pipeline can train and evaluate a PPO agent end to end, but this study provides no evidence that PPO learned to trade Henry Hub gas profitably."),
    ("p", "Objective iv was partially achieved. Experiment tracking was the strongest part of the system: every run stored its parameters, metrics, configuration and models, and four months after training these records were enough to rebuild the exact dataset and reproduce the logged validation returns (TC-09). The same records made it possible to locate defects DEF-01 to DEF-03; without them, the +124.02 % backtest could easily have been reported as a success. The model registry and the quality gate were not implemented, and the Airflow DAG was defined but never run (TC-14). Research question iv is therefore answered only at the design level: the DAG passes the MLflow run ID and the backtest summary between tasks, but this integration has not been shown to work under Airflow."),
    ("p", "Objective v was partially achieved. All 61 unit tests passed with 62 % statement coverage of the source code, the live EIA client and the full data-to-backtest path were tested, the Docker Compose file was validated (TC-15) and the held-out results were reproduced. Research question v is answered only in part. The most important lesson from testing is that a passing unit suite did not show the system was correct: the critical reward defect was found by investigating why the agent stopped trading and was then confirmed by a white-box test of the reward (TC-11), after all 61 unit tests had passed. Nine defects were recorded in total. The DAG, the Terraform configuration and the CI/CD workflows were not executed, so the orchestration, cloud and deployment requirements rest on configuration evidence only."),
    ("p", "Taken together, the aim was partially met. Ingestion, transformation, training, tracking and backtesting work and are reproducible; automated deployment and cloud provisioning exist as code only; and the forecasting module described in the proposal was replaced by an agent that outputs trading positions rather than price forecasts (Table 5.4)."),

    ("h3", ("6.2.2", "Contribution and Boundaries of the Findings")),
    ("p", "The project makes two contributions. The first is a tested and reproducible data-to-backtest pipeline for DRL trading on daily EIA energy prices, in which every reported result can be traced to a logged run and rebuilt from it. The second is the evaluation itself. It documents how a defect in the reward produced a backtest that looked far better than both baselines, and it shows which checks exposed the problem: reproducible tracking, a distribution of random agents rather than a single one, validation results read alongside test results, and a white-box test of the reward. This supports the position taken in Chapter 2 that tracking and reproducibility are as important to a working ML system as the model."),
    ("p", "These conclusions hold only within the conditions that were tested: daily Henry Hub prices from 2010 to 2026 with WTI as the only exogenous series; one held-out period of 414 trading days dominated by two price spikes; PPO trained once per configuration with seed 42 and no hyperparameter search; simulated execution at the published daily spot price with a fixed 0.1 % transaction cost; and a single developer workstation. The system has not been used for real trading, and none of the results supports acting on its decisions without a human trader reviewing them."),

    ("h2", ("6.3", "Recommendations")),
    ("p", "The recommendations below follow from the results and limitations in Section 5.8 and the defect log in Table 5.7. Each names the audience responsible, the action, the evidence behind it and its priority. The essential recommendations are safeguards that should be completed before the system or its results are used further; the remaining recommendations are improvements."),

    ("h3", ("6.3.1", "Essential Recommendations")),
    ("p", "i.  Correct the reward before any further training (system developers; highest priority). The reward in the environment's step function should be calculated after moving to the next day's price, so that it includes the profit or loss of the position held, and TC-11 should be kept in the test suite as a regression test. Every agent reported in Chapter 5 was trained on a cost-only reward (DEF-01), so no trading result can be interpreted until this is fixed and the agents are retrained."),
    ("p", "ii.  Correct the evaluation defects before reporting new results (system developers). The backtest should run over the whole test split rather than stopping at the episode length (DEF-02), the drawdown logged during training should be calculated on portfolio values (DEF-03), and the Random Agent's action sampler should be seeded (DEF-09). These defects caused the original backtest to cover only 252 of the 414 test days, a logged drawdown of -64,450,496.7 %, and random baselines that change between executions."),
    ("p", "iii.  Do not use the agent's outputs for trading decisions (prospective users such as energy traders and analysts). The agent should be treated as a research prototype that has been tested in simulation only. Its best test return came from a single price spike, it lost money on the validation period and it was trained without a profit signal. Any later version should keep a human trader responsible for every decision until it has been evaluated as described in Section 6.4."),
    ("p", "iv.  Pin the software versions that were actually used (system maintainers). requirements.txt should be updated to the versions in Table 5.2 (DEF-07). The results were produced with different versions from those pinned, and reproducibility (NFR-01) depends on the software environment as well as on the MLflow records."),

    ("h3", ("6.3.2", "Further Recommendations")),
    ("p", "v.  Execute the orchestration and deployment code before relying on it (system maintainers). The DAG should import EmptyOperator instead of the deprecated DummyOperator (DEF-04) and then be run under Airflow, the GitHub Actions workflows should be run on the repository, and the Terraform validation should be completed on a machine where the AWS provider starts (TC-16). A local Airflow run and the CI workflows would supply the missing evidence without the cost of a paid AWS deployment."),
    ("p", "vi.  Base the planned quality gate on validation evidence and baselines (system developers). The gate proposed in Chapter 3, a mean reward above zero, could never be passed under the implemented reward, while a gate based on test return alone would have promoted PPO-300k on the strength of one price spike. When the gate is implemented, it should require a candidate model to beat Buy & Hold and a stated percentile of the random-agent distribution on the validation period, with the test period kept for final reporting only."),
    ("p", "vii.  Revise data preparation for spike-driven commodities (system developers). The outlier rule should flag, not replace, values that appear in the published source, because it removed five genuine Henry Hub spikes, including the true January 2026 peak. The constant calendar features and hourly feature names should be removed or renamed for daily data (DEF-06), and the statistics used for normalisation and outlier detection should be fitted on the training split only (DEF-08)."),
    ("p", "viii.  Report DRL trading results against a distribution, not a single run (researchers and students reusing this work). On this test period, the returns of 30 random agents had a standard deviation of 61.5 percentage points and a best result of +169.07 %, so a single backtest return says little on its own. Results should be reported on both the validation and test periods, against Buy & Hold and a seeded set of random agents, together with the share of return that came from the largest single price move."),
    ("p", "ix.  Update the documentation (system maintainers). The README still describes the original hourly synthetic setup and should be rewritten for the daily EIA configuration, so that a new user can follow it to reproduce the results in Chapter 5."),
    ("p", "x.  No hardware change is recommended (adopting organisation). The 300,000-step training run took 14 minutes 57 seconds on a CPU, well within the two-hour limit of NFR-02, and the measurements show no compute, memory or storage bottleneck that would justify a GPU or cloud training."),

    ("h2", ("6.4", "Future Works")),
    ("p", "Future work should extend the study once the essential recommendations in Section 6.3.1 are complete; correcting the reward and evaluation defects is repair rather than future work. The directions below are listed in order of priority, and each arises from a limitation or open question in Section 5.8."),
    ("p", "i.  A controlled comparison of reward designs. Research question iii remains open because no agent was trained with a profit signal. With the corrected environment, PPO should be trained with at least two reward designs, for example profit and loss after costs alone, and profit and loss with the idle and drawdown penalties proposed in Chapter 3, using the same data, splits and hyperparameters so that only the reward changes. Each design should be trained with several seeds and judged on validation and test return, maximum drawdown and the share of random agents it beats. This would show whether the reward shaping discussed in Chapter 2 produces a real trading skill on this market."),
    ("p", "ii.  Robustness across seeds and time periods. The 414-day test period was dominated by two price spikes and each configuration was trained only once, so the variability of the results is unknown. A walk-forward evaluation, in which the agent is retrained and tested on successive periods of the 2010 to 2026 data with five or more seeds per period, would show whether a result depends on a few days. It needs no new data, only more training time, and success would be a return above Buy & Hold in most periods with a spread across seeds that is small compared with that margin."),
    ("p", "iii.  More realistic trade execution. Trades were simulated at the same day's published price with a fixed 0.1 % cost. Executing at the next day's price, adding a bid-ask spread and limiting position size relative to market volume would test whether any advantage survives realistic trading conditions. This requires spread and volume estimates for Henry Hub, which are not in the EIA series used here, and would be evaluated by repeating the comparison under each cost assumption and reporting how much of the return remains."),
    ("p", "iv.  Other markets and the original hourly setting. The agent has been evaluated only on Henry Hub gas. The ingestion module already takes the commodity as a parameter and the transformation module detects hourly or daily data, so the study could be repeated with WTI crude oil as the target and, if a suitable EIA route is found, with hourly electricity prices as originally proposed. This would be an external validation of whether conclusions drawn on one market carry over to another."),
    ("p", "v.  A sustained deployment study. Once the DAG has been executed and the quality gate implemented (Section 6.3.2), the pipeline could run daily on new EIA data for several months. This would measure what the present study could not: how often scheduled runs succeed, how often retrained models pass the gate, and whether the input data drift away from the training distribution. It would need a long-running Airflow installation or a funded cloud account, and would be judged by the run success rate, the gate decisions over time and a drift measure on the main features."),
    ("p", "In summary, the project delivered a reproducible data and experiment pipeline whose own records revealed that its trading agent had not been trained to trade. The most valuable next step is therefore not a larger model or more infrastructure, but a corrected reward evaluated under the controlled, multi-seed comparison described above."),
]


def strip(p):
    """Remove bookmarks and rendered-page-break markers from a cloned paragraph."""
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for el in p.findall(qn(tag)):
            p.remove(el)
    for el in p.iter(qn("w:lastRenderedPageBreak")):
        el.getparent().remove(el)
    return p


def set_runs(p, texts):
    """Replace the paragraph's runs: texts is a list of (has_tab, text)."""
    runs = p.findall(qn("w:r"))
    template = copy.deepcopy(runs[0])
    for r in runs:
        p.remove(r)
    for tab, text in texts:
        r = copy.deepcopy(template)
        for child in list(r):
            if child.tag != qn("w:rPr"):
                r.remove(child)
        if tab:
            r.append(r.makeelement(qn("w:tab"), {}))
        t = r.makeelement(qn("w:t"), {})
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        p.append(r)
    return p


def main():
    d = docx.Document(DOC)
    paras = d.paragraphs
    if any(p.text.startswith("Chapter Six") for p in paras):
        sys.exit("Chapter Six already exists; not inserting again.")

    def find(pred, what):
        for i, p in enumerate(paras):
            if pred(p):
                return i
        sys.exit(f"Template not found: {what}")

    h1_i = find(lambda p: p.text.startswith("Chapter Five:"), "Chapter Five heading")
    brk_i = h1_i - 1
    assert "w:br" in paras[brk_i]._p.xml and not paras[brk_i].text, "page break before Ch5"
    h2_i = find(lambda p: p.text == "5.1\tIntroduction", "5.1 heading")
    h3_i = find(lambda p: p.style.name == "Heading 3" and p.text.startswith("5.2.1"), "5.2.1")
    body_i = h2_i + 1
    anchor_i = find(lambda p: p.text.startswith("The conclusions, recommendations and future work"),
                    "end of 5.8")

    tpl = {k: paras[i]._p for k, i in
           dict(brk=brk_i, h1=h1_i, h2=h2_i, h3=h3_i, p=body_i).items()}

    new = [strip(copy.deepcopy(tpl["brk"]))]
    new.append(set_runs(strip(copy.deepcopy(tpl["h1"])), [(False, H1)]))
    for kind, val in CONTENT:
        el = strip(copy.deepcopy(tpl[kind]))
        if kind in ("h2", "h3"):
            num, title = val
            set_runs(el, [(False, num), (True, title)])
        else:
            set_runs(el, [(False, val)])
        new.append(el)

    cur = paras[anchor_i]._p
    for el in new:
        cur.addnext(el)
        cur = el
    d.save(DOC)
    print(f"Inserted {len(new)} paragraphs after paragraph {anchor_i}.")


if __name__ == "__main__":
    main()
