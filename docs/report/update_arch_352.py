"""One-shot (24 Nov 2026): bring Section 3.5.2 (System Architecture) in line with Figure 4.5.

Adds the model lifecycle layer (six layers, not five), the closed-loop DAG tasks and the
local Docker Compose stack (LocalStack in place of S3, PostgreSQL, MLflow server, Mailpit).
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_arch_352.docx
Run from the repo root:  python docs/report/update_arch_352.py
"""
import sys
from pathlib import Path

from docx import Document

DOC = Path(__file__).resolve().parents[2] / "energy_trading_mlops_proposal_v17.docx"

NEW = [
    "The system architecture is presented in Figure 4.5 in Chapter 4. It follows a layered design: six "
    "horizontal layers describe the flow of data through the platform, from the data sources to the model "
    "lifecycle and delivery, and a vertical platform layer provides the supporting infrastructure and DevOps "
    "services.",

    "The data sources layer contains the EIA API v2, which supplies daily Henry Hub natural gas and WTI crude "
    "oil prices, and a seeded synthetic data generator used for testing. The ingestion and storage layer writes "
    "raw and processed Parquet files to separate zones of an S3 data lake; in the implemented system these zones "
    "are emulated locally by LocalStack, and the AWS Glue crawlers that would catalogue them are defined in code "
    "but not provisioned. The processing and machine learning layer cleans and validates the data, engineers 38 "
    "features, wraps the result in the Gymnasium trading environment, trains the PPO agent with "
    "Stable-Baselines3 and logs every run to MLflow. The model lifecycle layer, added in November 2026, closes "
    "the loop: a quality gate tests each trained agent on the validation split against Buy & Hold and seeded "
    "Random Agents, passing agents are registered in the MLflow Model Registry and the best becomes the "
    "champion, the champion is paper traded (simulated) on new daily data, a monitor compares its performance "
    "with Buy & Hold, and a retraining step trains challengers when the monitor reports a breach, promoting one "
    "only if it passes the gate and beats the champion.",

    "The orchestration layer contains the Airflow DAG and the local run scripts that call the same modules "
    "directly. The DAG runs daily: its pipeline tasks ingest, transform, train, backtest and notify, and its "
    "closed-loop tasks paper trade, monitor and, when retraining is required, train challengers, evaluate and "
    "promote them and send an alert. The evaluation and delivery layer produces the backtest report against Buy "
    "& Hold and Random baselines, the offline trade-replay application and the email notifications. Alongside "
    "these, the platform layer provides Docker Compose for the local container stack, which runs the Airflow "
    "webserver and scheduler with a PostgreSQL metadata database, an MLflow tracking server, LocalStack in place "
    "of S3 and a Mailpit inbox for the emails; Terraform for AWS provisioning in us-east-1; GitHub Actions for "
    "CI/CD; CloudWatch and SNS for logging and alerts; and ECR and IAM for container images and access control. "
    "The diagram distinguishes the components used for the reported results, those that ran only in the local "
    "Docker Compose stack or in GitHub Actions, and those that are defined in code but not deployed.",
]

d = Document(str(DOC))
ps = d.paragraphs
i = next(k for k, p in enumerate(ps) if p.style.name == "Heading 3" and p.text.startswith("3.5.2"))
body = ps[i + 1:i + 4]
if body[0].text.startswith(NEW[0][:120]):
    sys.exit("Section 3.5.2 already updated")
assert body[0].text.startswith("The system architecture is presented in Figure 4.5"), body[0].text
assert body[1].text.startswith("The data sources layer"), body[1].text
assert body[2].text.startswith("The orchestration layer"), body[2].text
assert ps[i + 4].text.startswith("3.5.3"), ps[i + 4].text
for p, text in zip(body, NEW):
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
d.save(str(DOC))
print("Section 3.5.2 updated")
