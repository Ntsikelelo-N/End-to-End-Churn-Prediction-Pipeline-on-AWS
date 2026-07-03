"""
churn_pipeline — End-to-End Churn Prediction Pipeline on AWS.

Public API surface — import from here rather than from submodules
to insulate notebooks and scripts from internal restructuring.

Usage:
    from churn_pipeline import run_cleaning_pipeline, engineer_features
    from churn_pipeline import compare_models, evaluate_on_test_set
"""

import logging

from churn_pipeline.evaluate import (
    dummy_roc_auc_baseline,
    evaluate_on_test_set,
    get_feature_importance,
    majority_class_metrics,
    threshold_sweep,
)
from churn_pipeline.features import (
    build_preprocessor,
    engineer_features,
    make_train_test_split,
    split_features_target,
)
from churn_pipeline.ingest import (
    download_raw_data,
    load_raw_dataframe,
    read_csv_from_s3,
    upload_to_s3,
)
from churn_pipeline.preprocess import run_cleaning_pipeline
from churn_pipeline.train import (
    compare_models,
    fit_final_model,
    load_model,
    save_model,
)

# Configure a package-level logger so the caller decides the handler
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # ingest
    "download_raw_data",
    "load_raw_dataframe",
    "upload_to_s3",
    "read_csv_from_s3",
    # preprocess
    "run_cleaning_pipeline",
    # features
    "engineer_features",
    "build_preprocessor",
    "split_features_target",
    "make_train_test_split",
    # train
    "compare_models",
    "fit_final_model",
    "save_model",
    "load_model",
    # evaluate
    "evaluate_on_test_set",
    "threshold_sweep",
    "get_feature_importance",
    "dummy_roc_auc_baseline",
    "majority_class_metrics",
]
