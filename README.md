# Disk Failure Prediction in Cloud Data Centres

## MSA University — Cloud Computing Project 7

### Supervised by
Dr. Rabab

### Team Members
- Mohammed Hossam — 247713
- Ahmed Tamer — 244749
- Ahmed Mohamed — 242679
- Hamza Belal — 243387

## Project Description
This project predicts hard disk failures up to seven
days in advance using SMART attributes from the
Backblaze Drive Stats dataset.

The project uses Amazon Web Services (AWS) for
cloud storage, machine learning training,
experimentation, and monitoring.

## Technologies
- Python
- Amazon EC2
- Amazon S3
- AWS IAM
- Amazon CloudWatch
- Scikit-learn
- LightGBM

## Machine Learning Models
1. Logistic Regression (Baseline)
2. LightGBM Static Weighted
3. LightGBM Temporal Unweighted
4. LightGBM Temporal Weighted

## Dataset
Backblaze Drive Stats — Q1 and Q2 2023

## Experimental Results
The models were trained using Q1 2023 data
and evaluated on Q2 2023 data.

Static Weighted LightGBM achieved the lowest
false-positive rate in Q2 (36.35%).

The false-positive rates remain high, and
further improvements are necessary.

## Complete Source Code

[Open Complete Project](Project_7_MSA_University_GitHub_Source_REVIEW/disk-failure-prediction-cloud/)

The project folder contains the Python source code,
AWS configuration, experiments, architecture diagram,
documentation, and reproduction instructions.

## University
MSA University, Egypt
