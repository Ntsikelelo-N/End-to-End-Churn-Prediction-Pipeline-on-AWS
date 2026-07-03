"""
glue/churn_etl_job.py — AWS Glue PySpark ETL job for the churn pipeline.

CHANGED: The Glue Visual ETL was a point-and-click black box with no version
control. This script replaces it with auditable PySpark that mirrors the logic
in preprocess.py. Keeping the same business rules in both places (Glue for
batch ETL, Python for notebook) means cleaned data from S3 is always consistent
with what the model was trained on.

Deploy: paste this content into the Glue Script tab of your 'churn-etl-job',
or upload to S3 and reference it from the job definition in your IaC.

Glue version: 4.0 | Language: Python 3 | Worker type: G.1X | Workers: 2
"""

import sys
import logging

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# ---------------------------------------------------------------------------
# Bootstrap Glue context
# ---------------------------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "SOURCE_BUCKET", "SOURCE_KEY", "DEST_BUCKET", "DEST_PREFIX"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Parameters (injected at job runtime — no hardcoded bucket names)
# ---------------------------------------------------------------------------

SOURCE_PATH = f"s3://{args['SOURCE_BUCKET']}/{args['SOURCE_KEY']}"
DEST_PATH = f"s3://{args['DEST_BUCKET']}/{args['DEST_PREFIX']}"

# ---------------------------------------------------------------------------
# Step 1: Read raw CSV from S3
# ---------------------------------------------------------------------------

logger.info("Reading raw data from %s", SOURCE_PATH)
raw_df = spark.read.option("header", "true").option("inferSchema", "true").csv(SOURCE_PATH)

logger.info("Raw data schema:")
raw_df.printSchema()
logger.info("Row count: %d", raw_df.count())

# ---------------------------------------------------------------------------
# Step 2: Fix TotalCharges — blank strings to null, cast to double
# ---------------------------------------------------------------------------

cleaned = raw_df.withColumn(
    "TotalCharges",
    F.when(F.trim(F.col("TotalCharges")) == "", None).otherwise(F.col("TotalCharges")),
).withColumn(
    "TotalCharges",
    F.col("TotalCharges").cast(DoubleType()),
)

# Impute missing TotalCharges with MonthlyCharges (brand-new customers, tenure=0)
cleaned = cleaned.withColumn(
    "TotalCharges",
    F.when(F.col("TotalCharges").isNull(), F.col("MonthlyCharges")).otherwise(
        F.col("TotalCharges")
    ),
)

missing_after = cleaned.filter(F.col("TotalCharges").isNull()).count()
logger.info("Missing TotalCharges after imputation: %d", missing_after)

# ---------------------------------------------------------------------------
# Step 3: Encode binary Yes/No columns to integers
# ---------------------------------------------------------------------------

BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

for col_name in BINARY_YES_NO_COLS:
    cleaned = cleaned.withColumn(
        col_name,
        F.when(F.col(col_name) == "Yes", 1)
         .when(F.col(col_name).isin("No", "No phone service", "No internet service"), 0)
         .otherwise(None)
         .cast(IntegerType()),
    )

# ---------------------------------------------------------------------------
# Step 4: Encode target column (Churn: Yes=1, No=0)
# ---------------------------------------------------------------------------

cleaned = cleaned.withColumn(
    "Churn",
    F.when(F.col("Churn") == "Yes", 1).otherwise(0).cast(IntegerType()),
)

# ---------------------------------------------------------------------------
# Step 5: Encode gender (Male=1, Female=0)
# ---------------------------------------------------------------------------

cleaned = cleaned.withColumn(
    "gender",
    F.when(F.col("gender") == "Male", 1).otherwise(0).cast(IntegerType()),
)

# ---------------------------------------------------------------------------
# Step 6: Drop identity column
# ---------------------------------------------------------------------------

cleaned = cleaned.drop("customerID")

# ---------------------------------------------------------------------------
# Step 7: Derived features (mirrors features.py logic)
# ---------------------------------------------------------------------------

cleaned = cleaned.withColumn(
    "AvgMonthlyCharges",
    F.when(
        F.col("tenure") > 0,
        F.col("TotalCharges") / F.col("tenure"),
    ).otherwise(F.col("MonthlyCharges")),
)

cleaned = cleaned.withColumn(
    "ServiceCount",
    (
        F.col("OnlineSecurity")
        + F.col("OnlineBackup")
        + F.col("DeviceProtection")
        + F.col("TechSupport")
        + F.col("StreamingTV")
        + F.col("StreamingMovies")
    ).cast(IntegerType()),
)

# ---------------------------------------------------------------------------
# Step 8: Write cleaned Parquet to S3
# ---------------------------------------------------------------------------

logger.info("Writing cleaned data to %s", DEST_PATH)
cleaned.write.mode("overwrite").parquet(DEST_PATH)
logger.info("ETL job complete. Output row count: %d", cleaned.count())

job.commit()
