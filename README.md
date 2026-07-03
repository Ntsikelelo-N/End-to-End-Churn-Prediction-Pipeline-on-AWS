# End-to-End Customer Churn Prediction Pipeline on AWS

[![CI](https://github.com/Ntsikelelo-N/End-to-End-Churn-Prediction-Pipeline-on-AWS/actions/workflows/ci.yml/badge.svg)](https://github.com/Ntsikelelo-N/End-to-End-Churn-Prediction-Pipeline-on-AWS/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-pattern churn prediction pipeline built on AWS Free Tier services. Raw telco data flows from S3 → Glue ETL → cleaned Parquet → a scikit-learn Pipeline trained locally and evaluated against a meaningful baseline. All business logic lives in an installable Python package (`src/churn_pipeline`), not in notebook cells.

---

## Results

| Model | ROC-AUC (5-fold CV) | F1 (churn class) |
|---|---|---|
| Logistic Regression | 0.843 ± 0.012 | 0.621 |
| Random Forest | 0.856 ± 0.009 | 0.637 |
| **XGBoost** | **0.864 ± 0.008** | **0.651** |
| Dummy baseline (majority class) | 0.500 | 0.000 |

> Predicting "no churn" for every customer achieves **74% accuracy but 0% recall** on the churn class — the models above are compared against this honest baseline, not raw accuracy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Free Tier                           │
│                                                                 │
│  IBM Telco CSV ──► S3 (raw/)                                   │
│                        │                                        │
│                    Glue Crawler                                 │
│                    (catalogues schema in Glue Data Catalog)    │
│                        │                                        │
│                    Glue ETL Job (PySpark)                       │
│                    glue/churn_etl_job.py                        │
│                        │                                        │
│                    S3 (cleaned/ — Parquet)                      │
└─────────────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼────────────┐
              │  Local / Notebook      │
              │                        │
              │  churn_pipeline.ingest │  ◄── reads from S3
              │  churn_pipeline.       │
              │    preprocess          │  ◄── type fixing, encoding
              │  churn_pipeline.       │
              │    features            │  ◄── feature engineering
              │  churn_pipeline.train  │  ◄── CV model selection
              │  churn_pipeline.       │
              │    evaluate            │  ◄── ROC-AUC, threshold analysis
              └────────────────────────┘
```

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint + test on every push
├── data/
│   ├── raw/                    # Downloaded CSV (git-ignored)
│   └── processed/              # Feature-engineered output (git-ignored)
├── glue/
│   └── churn_etl_job.py        # PySpark ETL replacing the Visual ETL job
├── models/                     # Saved model artefacts (git-ignored)
├── notebooks/
│   └── 01_eda_and_modelling.ipynb
├── screenshots/                # AWS console setup screenshots
├── src/
│   └── churn_pipeline/
│       ├── __init__.py         # Public API surface
│       ├── config.py           # Centralised config (no magic numbers)
│       ├── ingest.py           # Download + S3 upload/download
│       ├── preprocess.py       # Data cleaning (dtype fixes, encoding)
│       ├── features.py         # Feature engineering + ColumnTransformer
│       ├── train.py            # Pipeline building, CV, model persistence
│       └── evaluate.py         # Metrics, threshold sweep, feature importance
├── tests/
│   ├── test_preprocess.py      # 20 unit tests for cleaning functions
│   ├── test_features.py        # 18 unit tests for feature engineering
│   └── test_train.py           # 15 unit tests for train + evaluate
├── .gitignore
├── Makefile
├── README.md
├── requirements.txt
└── setup.py
```

---

## Quick Start (Local)

**Prerequisites:** Python 3.10+, AWS CLI configured (`aws configure`)

```bash
# 1. Clone and install
git clone https://github.com/Ntsikelelo-N/End-to-End-Churn-Prediction-Pipeline-on-AWS.git
cd End-to-End-Churn-Prediction-Pipeline-on-AWS
pip install -e ".[dev,ml]"

# 2. Download the dataset
make download

# 3. Run the test suite
make test

# 4. Open the notebook
jupyter notebook notebooks/01_eda_and_modelling.ipynb
```

---

## AWS Setup (One-Time)

> Full console screenshots are in the [`screenshots/`](screenshots/) directory.

### Step 1 — IAM User

1. IAM → Users → **Create user** (attach `AdministratorAccess`)
2. Create an **Access Key** (CLI type) and download the CSV
3. Run `aws configure` and paste in your key, secret, and region (`us-east-1`)

### Step 2 — S3 Bucket

```bash
aws s3 mb s3://churn-project-<your-name> --region us-east-1
```

### Step 3 — Upload raw data

```python
from churn_pipeline import download_raw_data, upload_to_s3

download_raw_data()
upload_to_s3("data/raw/Telco-Customer-Churn.csv", "raw_data/Telco-Customer-Churn.csv")
```

### Step 4 — Glue Crawler

1. Glue → Crawlers → **Create crawler** (`churn-raw-crawler`)
2. Data source: `s3://churn-project-<your-name>/raw_data/`
3. IAM role: create `AWSGlueChurnRole` with `AWSGlueServiceRole` + `AmazonS3FullAccess`
4. Output database: `churn_db`, table prefix: `raw_`
5. **Run** the crawler

### Step 5 — Glue ETL Job

1. Glue → Jobs → **Visual ETL** → switch to **Script** tab
2. Paste the contents of [`glue/churn_etl_job.py`](glue/churn_etl_job.py)
3. Job details: `AWSGlueChurnRole`, Glue 4.0, G.1X, 2 workers
4. Add job parameters:
   - `--SOURCE_BUCKET` → `churn-project-<your-name>`
   - `--SOURCE_KEY` → `raw_data/Telco-Customer-Churn.csv`
   - `--DEST_BUCKET` → `churn-project-<your-name>`
   - `--DEST_PREFIX` → `cleaned_data/`
5. **Run** the job

Verify the output:
```bash
aws s3 ls s3://churn-project-<your-name>/cleaned_data/
```

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features, ~26% positive churn rate.

| Feature type | Examples |
|---|---|
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Binary | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` |
| Categorical | `Contract`, `InternetService`, `PaymentMethod` |
| Target | `Churn` (1 = churned, 0 = retained) |

**Known quirk:** 11 rows have blank `TotalCharges` — these are new customers (tenure=0) who have not yet been billed. Imputed with `MonthlyCharges`.

Source: [IBM via scottdangelo/GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

---

## Development

```bash
make lint      # flake8 + isort check
make format    # black + isort auto-fix
make test      # pytest + coverage report
make clean     # remove __pycache__, .egg-info, coverage artefacts
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Storage | Amazon S3 |
| Cataloguing | AWS Glue Data Catalog + Crawler |
| ETL | AWS Glue (PySpark 4.0) |
| ML framework | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |
| Language | Python 3.10+ |

---

## Author

**Ntsikelelo Jantjie** — Data Science Practitioner, Johannesburg  
[GitHub](https://github.com/Ntsikelelo-N) · [Portfolio](https://ntsikelelo-n.github.io)
