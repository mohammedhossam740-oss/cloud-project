# AWS authorization and cost checklist (review BEFORE creating resources)

- [ ] I am authorized to use this AWS account and incur costs.
- [ ] I selected a region and checked current on-demand pricing in AWS Pricing Calculator.
- [ ] I checked expected EC2 hourly price (example compute profile: t3.xlarge, 4 vCPU/16 GiB).
- [ ] I checked EBS volume size and charges (recommend >=30 GiB gp3; ZIP isn't extracted).
- [ ] I checked S3 storage cost and potential request and transfer costs.
- [ ] I checked CloudWatch Agent ingestion/retention and metrics charges.
- [ ] I set AWS Budgets or a billing alert before starting and noted that alerts do not stop instances.
- [ ] I attached a least-privilege IAM role to EC2; no embedded access keys.
- [ ] I restrict SSH ingress to my IP; or use authorized Session Manager where configured.
- [ ] I will stop or terminate EC2 after experiments and delete unneeded EBS/logs/S3 data.

Pricing varies by region, dates and instance type. This repository does **not** automatically create paid resources.
Pricing Calculator: https://calculator.aws/
