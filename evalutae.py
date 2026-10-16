
import os
import json
import pickle
import pandas as pd
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# MLflow tracking configuration
MLFLOW_TRACKING_URI = (
    "https://dagshub.com/akshaygowda52130/mlpipeline.mlflow"
)

# Existing files
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_model():
    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("akshaygowda52130")

    # Load existing dataset
    df = pd.read_csv(DATA_PATH)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Ensure labels match the model's numeric classes
    y = pd.to_numeric(y).astype(int)

    # Recreate the same test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Load the existing trained model — no training
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Existing model loaded successfully.")
    print("Model path:", MODEL_PATH)

    # Predict using the loaded model
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(int).to_numpy()

    # Calculate evaluation metrics
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_weighted": float(
            precision_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "test_recall_weighted": float(
            recall_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "test_f1_weighted": float(
            f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )
        )
    }

    # Log results in a separate MLflow run
    with mlflow.start_run(run_name="model_evaluation") as run:
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("evaluation_dataset", DATA_PATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metrics(metrics)

        # Classification report
        report = classification_report(
            y_test, y_pred, zero_division=0
        )
        mlflow.log_text(
            report,
            "evaluation/classification_report.txt"
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)

        mlflow.log_text(
            cm_df.to_csv(index=False),
            "evaluation/confusion_matrix.csv"
        )

        # Metrics JSON artifact
        mlflow.log_text(
            json.dumps(metrics, indent=4),
            "evaluation/metrics.json"
        )

        # Display results
        print("\nEvaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nMLflow Run ID:", run.info.run_id)
        print(
            "MLflow Run URL: "
            f"{MLFLOW_TRACKING_URI}/#/experiments/0/runs/"
            f"{run.info.run_id}"
        )


if __name__ == "__main__":
    evaluate_model()
