#!/usr/bin/env bash
set -euo pipefail
# Run only when AWS EC2 instance has the project role with S3 bucket permissions.
: "${DISK_BUCKET:?Export DISK_BUCKET=your-existing-unique-bucket-name}"
: "${AWS_DEFAULT_REGION:?Export AWS_DEFAULT_REGION=your-AWS-region}"
PREFIX="s3://${DISK_BUCKET}/disk-failure"
# Fail before downloading/training if the EC2 IAM role or bucket is not set up.
command -v aws >/dev/null 2>&1 || { echo "AWS CLI missing: run bash aws/setup_ubuntu_ec2.sh" >&2; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "No AWS identity found: attach an IAM role to this EC2 instance" >&2; exit 1; }
aws s3api head-bucket --bucket "${DISK_BUCKET}" >/dev/null || { echo "Cannot access S3 bucket ${DISK_BUCKET}; check region, name, and IAM role" >&2; exit 1; }
source .venv/bin/activate
mkdir -p data/raw data/processed results/metrics results/figures models
ARCHIVE=$(python - <<'PY'
import yaml
print(yaml.safe_load(open('config/config.yaml'))['archive_name'])
PY
)
# Dataset is first fetched onto EC2, saved in S3, and reloaded from S3 to
# explicitly validate the cloud-storage leg of the architecture.
if ! aws s3api head-object --bucket "${DISK_BUCKET}" --key "disk-failure/raw/${ARCHIVE}" >/dev/null 2>&1; then
  python -m src.cli download
  aws s3 cp "data/raw/${ARCHIVE}" "${PREFIX}/raw/${ARCHIVE}"
fi
aws s3 cp "${PREFIX}/raw/${ARCHIVE}" "data/raw/${ARCHIVE}"
{
  printf '\n====== CLOUD RUN START %s ======\n' "$(date -u)"
  echo "Instance: $(hostname), AWS region: ${AWS_DEFAULT_REGION}"
  python -m src.cli prepare
  aws s3 cp data/processed/drive_features.parquet "${PREFIX}/processed/drive_features.parquet"
  aws s3 cp data/processed/dataset_summary.json "${PREFIX}/processed/dataset_summary.json"
  python -m src.cli eda
  python -m src.cli train
  # Holdout is a secondary experiment. It may fail (e.g. only one positive class)
  # on a small period; never overwrite or invent standard test results.
  python -m src.cli train --holdout-model ST14000NM001G || \
    echo 'Unseen-model experiment unavailable with this data; see dataset class counts.'
  printf '\n====== CLOUD RUN END %s ======\n' "$(date -u)"
} 2>&1 | tee -a cloud_experiment.log
aws s3 sync data/processed "${PREFIX}/processed" --exclude '*.pkl'
aws s3 sync models "${PREFIX}/models" --exclude '*.pkl'
aws s3 sync results "${PREFIX}/results"
if [ -d holdout_runs ]; then
  aws s3 sync holdout_runs "${PREFIX}/holdout_runs"
fi
aws s3 cp cloud_experiment.log "${PREFIX}/logs/cloud_experiment.log"
echo "Cloud-run artifacts uploaded to ${PREFIX}"
