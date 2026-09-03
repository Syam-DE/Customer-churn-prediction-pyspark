"""
End-to-end PySpark pipeline for customer churn prediction.

Steps:
  1. Load raw customer CSV into a Spark DataFrame
  2. Feature engineering: encode categoricals, assemble feature vector
  3. Train/test split
  4. Train a Random Forest classifier (PySpark ML)
  5. Evaluate: accuracy, AUC, precision/recall, feature importance
  6. Write predictions + metrics to output/

Run: python src/churn_pipeline.py
"""

import json
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

DATA_PATH = "data/customers.csv"
OUTPUT_DIR = "output"

CATEGORICAL_COLS = ["plan_type", "acquisition_channel", "loyalty_tier"]
NUMERIC_COLS = [
    "tenure_months",
    "weekly_deliveries_last_8w",
    "skipped_weeks_last_8w",
    "avg_box_price",
    "discount_pct_last_order",
    "support_tickets_last_90d",
    "avg_ticket_resolution_hrs",
    "late_deliveries_last_90d",
    "app_sessions_last_30d",
    "recipe_swaps_last_30d",
    "days_since_last_login",
    "referred_friends_total",
]
LABEL_COL = "churned"


def build_spark_session():
    return (
        SparkSession.builder.appName("CustomerChurnPrediction")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def build_feature_pipeline():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in CATEGORICAL_COLS
    ]
    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + [f"{c}_ohe" for c in CATEGORICAL_COLS],
        outputCol="features",
    )
    rf = RandomForestClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        numTrees=150,
        maxDepth=8,
        seed=42,
    )
    return Pipeline(stages=indexers + encoders + [assembler, rf])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    print(f"Loaded {df.count()} rows, {len(df.columns)} columns")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count()}  Test: {test_df.count()}")

    pipeline = build_feature_pipeline()
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    # --- Evaluation ---
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
    )
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedRecall"
    )

    metrics = {
        "accuracy": acc_evaluator.evaluate(predictions),
        "auc": auc_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "weighted_precision": precision_evaluator.evaluate(predictions),
        "weighted_recall": recall_evaluator.evaluate(predictions),
        "test_rows": test_df.count(),
        "train_rows": train_df.count(),
    }

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Feature importance (numeric + one-hot expanded)
    rf_model = model.stages[-1]
    assembler_stage = [s for s in model.stages if isinstance(s, VectorAssembler)][0]
    feature_names = assembler_stage.getInputCols()
    importances = rf_model.featureImportances.toArray()

    # one-hot columns expand into multiple slots; approximate by aligning
    # numeric cols 1:1 and reporting categorical importance as a block sum
    numeric_importance = {
        feature_names[i]: float(importances[i]) for i in range(len(NUMERIC_COLS))
    }
    top_features = dict(
        sorted(numeric_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    )

    print("\n=== Top numeric feature importances ===")
    for k, v in top_features.items():
        print(f"{k}: {v:.4f}")

    # --- Persist outputs ---
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump({"metrics": metrics, "top_feature_importances": top_features}, f, indent=2)

    predictions.select(
        "customer_id", LABEL_COL, "prediction", "probability"
    ).toPandas().to_csv(f"{OUTPUT_DIR}/predictions_sample.csv", index=False)

    print(f"\nWrote metrics to {OUTPUT_DIR}/metrics.json")
    print(f"Wrote predictions to {OUTPUT_DIR}/predictions_sample.csv")

    spark.stop()


if __name__ == "__main__":
    main()
