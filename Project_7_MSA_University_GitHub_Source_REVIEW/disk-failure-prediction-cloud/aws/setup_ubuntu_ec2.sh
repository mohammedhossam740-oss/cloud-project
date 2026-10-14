#!/usr/bin/env bash
set -euo pipefail
# Ubuntu 24.04 LTS. Commands run ON your authorized EC2 instance.
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip unzip curl git
# Install the AWS CLI v2 if the selected Ubuntu AMI does not include it.
if ! command -v aws >/dev/null 2>&1; then
  curl -fsSL https://awscli.amazonaws.com/v2/install.sh -o /tmp/install-aws-cli.sh
  sudo bash /tmp/install-aws-cli.sh --system
fi
aws --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
echo 'EC2 environment ready. Run aws/run_cloud_experiments.sh from repository root.'
