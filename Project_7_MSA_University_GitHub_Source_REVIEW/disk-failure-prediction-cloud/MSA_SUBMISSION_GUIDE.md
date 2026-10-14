# MSA University - Project 7 | Submission and cloud reproduction

**Title:** Disk Failure Prediction in Cloud Data Centres Under Extreme Class Imbalance.

**Team:** Mohammed Hossam (247713); Ahmed Tamer (244749); Ahmed Mohamed (242679); Hamza Belal (243387).

**GitHub destination:** https://github.com/mohammedhossam740-oss/cloud-project
**Status:** Repository destination was empty when checked on 15 Oct 2026. Commit the code before submission, and note its commit hash.

## Observed AWS execution environment
- AWS region: `eu-north-1` (Stockholm); EC2 Amazon Linux 2023, t3.micro, Python 3.11; EBS gp3 20 GiB.
- Private S3 bucket `disk-failure-ml-project-2026`; IAM instance profile `DiskFailureEC2Role`; CloudWatch built-in CPUUtilization metric.
- Q1 ZIP and Q2 ZIP sourced from Backblaze Drive Stats (Q1/Q2 2023). Full ZIP files are deliberately excluded from this GitHub-ready package.
- `src/data.py` includes a needed fix to ignore macOS `__MACOSX/` and `._` ZIP metadata entries.

## Data and experiment workflow on EC2
From project root, install Python packages and run tests:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
python -m pytest -q
```

Download official Q1 from the Backblaze source via `python -m src.cli download`, and obtain `data/raw/data_Q2_2023.zip` from Backblaze. Upload to S3 as private objects. Do not commit raw datasets.

Q1 (outcome-enriched **pilot only**):
```bash
python -u low_ram_pilot.py
python -m src.cli prepare --config config/pilot.yaml --archive data/raw/pilot_Q1_2023.zip --parquet data/processed/pilot_features.parquet
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m src.cli train --config config/pilot.yaml --parquet data/processed/pilot_features.parquet
```

Q2 (later-period natural-prevalence evaluation of all eligible drive-days of two selected HDD models; frozen Q1 thresholds):
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u q2_independent_eval.py
```

**CAUTION:** `q2_independent_eval.py` refuses to overwrite existing `results/q2_independent/q2_predictions.csv`. Move/rename prior output before an intentional rerun, and document any rerun. No models are retrained on Q2.

Keep existing trained models `models/*.joblib` on private S3, if using a second machine: obtain the Q1-trained model artifacts before running Q2 evaluation.

## Results provided
- `results/pilot/comparison.csv` and `run_metadata.json`: Q1 pilot, not representative precision or alert burden.
- `results/q2_independent/comparison.csv` and `summary.json`: real later-period evaluation; 2,327,260 drive-days and 1,317 positive labels.
- `report/MSA_Project7_Report_REVIEW.docx` / `.pdf`: updated academic report with CloudWatch evidence and limitations.
- `presentation/MSA_Project7_Presentation_REVIEW.pptx`: 15-slide summary.

## Publication checklist
1. Do NOT upload AWS credentials, `.pem` keys, `.env`, S3 object copies, personal data, or complete Backblaze archives.
2. Commit this extracted folder's **contents**, not a ZIP file, to the GitHub repository.
3. Verify public GitHub shows `src/`, `aws/`, `README.md`, `low_ram_pilot.py`, and `q2_independent_eval.py`.
4. Record commit hash and professor/instructor name in report, then submit.
5. Validate project experiments further if possible: high false-positive rates mean no production readiness.
