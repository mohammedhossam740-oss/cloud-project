# Project 7: actual AWS execution (account already created)

**Status:** These instructions have been prepared, but the cloud deployment and real-data experiment have not been executed or verified. **AWS services may incur charges.** Before launching EC2, review the pricing calculator for your chosen region and approve the expense yourself. Never share passwords, access keys, private keys, or account identifiers.

## Step 1 — Budget (before provisioning)

AWS Console → **Billing and Cost Management → Budgets → Create budget**. Create a monthly cost budget with email alerts (for example USD $10, with 50% and 80% notifications). An alert is NOT a hard spending cap; charges can exceed it. Review the pricing of t3.xlarge + gp3 EBS + S3 + transfer + CloudWatch in your region and make sure you have sufficient credits or spending authorization.

## Step 2 — S3 bucket

Console → **S3 → Create bucket**. Choose **General purpose**, a globally unique name (e.g. `disk-failure-course-unique-suffix`), and ONE Region to use for EC2. Leave **Block all public access ON**, object ownership **Bucket owner enforced** and default encryption on. Keep note of just the bucket name and Region for your commands. Do not make the bucket public.

## Step 3 — EC2 IAM role (no API keys)

Console → **IAM → Policies → Create policy → JSON**. Open `aws/iam_ec2_s3_policy.template.json`, replace *every* `REPLACE_WITH_BUCKET_NAME` with your exact bucket name, paste as JSON and create a policy named `DiskFailureProjectS3Policy`.

Console → **IAM → Roles → Create role → AWS service → EC2**. Attach your newly created policy, and optionally the AWS-managed `CloudWatchAgentServerPolicy` for the CloudWatch Agent. Name it `DiskFailureProjectEC2Role`. The role and instance profile must be attached to the instance during launch, or through EC2 → Actions → Security → Modify IAM role. The `iam_ec2_trust_policy.json` in this project shows the EC2 trust relationship for reference.

## Step 4 — EC2 instance (creates billable infrastructure)

**Only after explicitly deciding to incur potential charges.** Console → EC2 → Launch instance:

- Name: `disk-failure-project7`
- AMI: **Ubuntu Server 24.04 LTS x86_64**.
- Instance type starting point: **t3.xlarge** (4 vCPU / 16 GiB RAM). Availability and exact price vary by region; CPU credits/unlimited mode can affect costs.
- EBS root disk: **50 GiB gp3** as a starting point; larger datasets may need more.
- **Advanced details → IAM instance profile:** choose `DiskFailureProjectEC2Role`.
- Security group: allow inbound SSH TCP 22 **from My IP only** if using SSH, not from the entire Internet. A public IPv4 address can also incur charges. Ensure outbound HTTPS traffic is allowed for dependencies and Backblaze download.
- Connect using a securely configured SSH key or EC2 Instance Connect. Never send the private SSH key to anyone.

## Step 5 — Upload and transfer the project ZIP (no GitHub required)

On **your laptop**, download `Project_7_Disk_Failure_Prediction_AWS.zip` from this conversation. In the **S3 console**, open your bucket, create folders `disk-failure/code/`, and upload the ZIP there. That upload is used only to transfer code; the experiment will separately upload the Backblaze data under `disk-failure/raw/`.

On the **EC2 terminal**, after connecting, run:

```bash
sudo apt-get update
sudo apt-get install -y unzip curl
curl -fsSL https://awscli.amazonaws.com/v2/install.sh -o /tmp/install-aws-cli.sh
sudo bash /tmp/install-aws-cli.sh --system
aws --version
export DISK_BUCKET="REPLACE_WITH_YOUR_BUCKET_NAME"
export AWS_DEFAULT_REGION="REPLACE_WITH_YOUR_REGION"
aws sts get-caller-identity    # verifies the EC2 IAM role; do NOT share output
aws s3 cp "s3://${DISK_BUCKET}/disk-failure/code/Project_7_Disk_Failure_Prediction_AWS.zip" .
unzip -q Project_7_Disk_Failure_Prediction_AWS.zip
cd disk-failure-prediction-cloud
bash aws/setup_ubuntu_ec2.sh
```

If you see an access-denied error, check the instance role and policy. Do not create permanent IAM access keys as a workaround.

## Step 6 — Execute real experiments on EC2

From the project folder, in the **same shell** with both `DISK_BUCKET` and `AWS_DEFAULT_REGION` exported:

```bash
bash aws/run_cloud_experiments.sh
aws s3 ls "s3://${DISK_BUCKET}/disk-failure/results/metrics/"
aws s3 cp "s3://${DISK_BUCKET}/disk-failure/results/metrics/comparison.csv" -
```

The script downloads Backblaze **2023 Q1 ZIP** on EC2 (~819 MB), uploads the ZIP to S3, downloads it from S3, creates 7-day labels, trains Logistic Regression and LightGBM, and uploads outputs. This dataset is ~7.53 GB extracted, but the code stream-reads CSV files directly from the archive; it does not fully unzip it. **If failure counts are too low, expand to additional quarters before making conclusions.** Training may need more RAM or disk as the selected data changes.

## Step 7 — Verify CloudWatch and preserve proof

EC2 has basic CloudWatch instance metrics by default. To collect the log file and RAM/disk metrics, install and configure **CloudWatch Agent** separately using the included `aws/cloudwatch_agent.json` (adjust its `file_path` if the project is stored elsewhere). Save evidence of EC2 instance details, the run timestamp and log, S3 raw data and result objects, and CloudWatch metrics. **Do not claim that CloudWatch log ingestion is active just because the configuration file exists.**

## Step 8 — Finish report, then clean up

Copy actual metrics in `comparison.csv`, figures and S3/EC2 evidence into the **DRAFT** report and slides; note real selected dates and failure counts. Run `python -m pytest -q` on EC2 and record the output.

After verifying result backups, stop or terminate EC2, and inspect EBS volumes, S3 storage, CloudWatch logs, any allocated IP address and Billing for continuing charges. A stopped instance can still incur storage and IP-related costs.

**Troubleshooting:** If you lack permission to create an IAM role, budget, or EC2 instance, your AWS account permissions may be restricted. Use an account administrator to grant narrowly scoped access rather than sharing their credentials.
