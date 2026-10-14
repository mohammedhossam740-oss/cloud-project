# Start here: Project 7

**Your project is built as a source-code-and-report draft. It is not yet a completed AWS experiment.**

### What you can do right away

1. Open `report/final_report_DRAFT_cloud_run_pending.pdf` to review the academic explanation.
2. Open `presentation/project7_presentation_DRAFT_results_pending.pptx` for your class presentation draft.
3. Study `diagrams/aws_architecture.png` to understand the cloud data flow.
4. Open `README.md` for full installation and cloud execution instructions.
5. Review the AWS cost checklist. AWS fees may apply and should be explicitly authorized by you before running any paid resources.

### How to finish the compulsory AWS requirement

Use your OWN AWS account and private S3 bucket with an EC2 instance role. On Ubuntu EC2, run:

```bash
bash aws/setup_ubuntu_ec2.sh
export DISK_BUCKET=YOUR_UNIQUE_BUCKET
export AWS_DEFAULT_REGION=YOUR_SELECTED_REGION
bash aws/run_cloud_experiments.sh
```

The script copies the original real Backblaze ZIP to S3, retrieves it, runs dataset preparation and experiments on EC2, then uploads results. These operations **have not** been performed in this authoring environment due to no access to the official ZIP and no AWS account access.

### Submission checklist

- [ ] Real Backblaze data processed (not the synthetic smoke fixture).
- [ ] Real EC2 experiment run, timestamp, instance type and AWS region recorded.
- [ ] S3 contains raw/processed/results objects; take evidence screenshot.
- [ ] AWS CloudWatch metrics/logs visible; take evidence screenshot.
- [ ] Actual comparison CSV / confusion matrices copied into the report and presentation.
- [ ] Experiment limitations and any splits with insufficient failing drives discussed.
- [ ] Team information and GitHub URL filled in; run `python -m pytest -q`.
- [ ] Unneeded AWS instances/volumes terminated and billing checked.

**Never upload AWS secret keys or passwords in chat or commit them to GitHub.**
