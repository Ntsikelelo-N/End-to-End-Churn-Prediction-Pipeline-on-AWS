"""
ingest.py — Data ingestion layer: download from source, push/pull to S3.

CHANGED: Notebook cells that downloaded the CSV and uploaded it to S3 are
extracted into pure functions with logging and error handling. This makes
the notebook thin and the logic reusable and testable.
"""

import logging
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import BotoCoreError, ClientError

from churn_pipeline.config import aws
from churn_pipeline.config import data as data_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source download
# ---------------------------------------------------------------------------


def download_raw_data(
    url: str = data_cfg.source_url,
    dest_path: str = data_cfg.local_raw_path,
    force: bool = False,
) -> Path:
    """Download the Telco Churn CSV from the IBM GitHub mirror.

    Args:
        url: Direct download URL for the CSV.
        dest_path: Local file path where the CSV should be saved.
        force: Re-download even if the file already exists.

    Returns:
        Path object pointing to the downloaded file.

    Raises:
        requests.HTTPError: If the HTTP response status is 4xx/5xx.
        IOError: If the file cannot be written to disk.
    """
    dest = Path(dest_path)
    if dest.exists() and not force:
        logger.info("Raw data already present at %s; skipping download.", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading raw data from %s …", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest.write_bytes(response.content)
    logger.info("Saved %d bytes to %s.", len(response.content), dest)
    return dest


def load_raw_dataframe(path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame without any transformation.

    Args:
        path: Override the default raw data path from config.

    Returns:
        Raw DataFrame with original column names and dtypes.
    """
    resolved = Path(path or data_cfg.local_raw_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {resolved}. "
            "Run download_raw_data() first or supply an explicit path."
        )
    df = pd.read_csv(resolved)
    logger.info(
        "Loaded raw data: %d rows × %d columns from %s.",
        len(df),
        df.shape[1],
        resolved,
    )
    return df


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client():
    """Return a boto3 S3 client using the configured region.

    Credentials are resolved in the standard boto3 order:
    environment variables → ~/.aws/credentials → IAM role.
    """
    return boto3.client("s3", region_name=aws.region)


def upload_to_s3(
    local_path: str,
    s3_key: str,
    bucket: str = aws.bucket_name,
) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Absolute or relative path to the local file.
        s3_key: S3 object key (path inside the bucket).
        bucket: Target S3 bucket name.

    Raises:
        FileNotFoundError: If local_path does not exist.
        ClientError: If the S3 upload fails (permissions, bucket missing, etc.).
    """
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"Cannot upload non-existent file: {src}")

    client = _get_s3_client()
    try:
        logger.info("Uploading %s → s3://%s/%s …", src, bucket, s3_key)
        client.upload_file(str(src), bucket, s3_key)
        logger.info("Upload complete.")
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 upload failed: %s", exc)
        raise


def download_from_s3(
    s3_key: str,
    local_path: str,
    bucket: str = aws.bucket_name,
) -> Path:
    """Download an S3 object to a local file.

    Args:
        s3_key: S3 object key.
        local_path: Destination file path.
        bucket: Source S3 bucket name.

    Returns:
        Path object pointing to the downloaded file.
    """
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    client = _get_s3_client()
    try:
        logger.info("Downloading s3://%s/%s → %s …", bucket, s3_key, dest)
        client.download_file(bucket, s3_key, str(dest))
        logger.info("Download complete.")
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 download failed: %s", exc)
        raise

    return dest


def read_csv_from_s3(s3_key: str, bucket: str = aws.bucket_name) -> pd.DataFrame:
    """Stream a CSV directly from S3 into a DataFrame (no local file written).

    Args:
        s3_key: S3 object key for the CSV file.
        bucket: Source S3 bucket name.

    Returns:
        DataFrame loaded from S3.
    """
    client = _get_s3_client()
    obj = client.get_object(Bucket=bucket, Key=s3_key)
    df = pd.read_csv(obj["Body"])
    logger.info(
        "Streamed %d rows × %d columns from s3://%s/%s.",
        len(df),
        df.shape[1],
        bucket,
        s3_key,
    )
    return df
