
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
import yaml
import os
import pickle
from mlflow.models import infer_signature
from urllib.parse import urlparse


# DagsHub MLflow tracking configuration
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/akshaygowda52130/mlpipeline.mlflow"
)

# Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# in your PowerShell environment. Never hardcode your token here.


def hyperparamter_tuning(x_train, y_train, param_grid, random_state):
    rf = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring="accuracy"
    )

    grid_search.fit(x_train, y_train)

    return grid_search


# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]


def train_model(data_path, model_path, random_state, test_size, param_grid):

    # Load the dataset
    df = pd.read_csv(data_path, header=None)

    if df.shape[1] < 2:
        raise ValueError(
            "The dataset must contain at least one feature column "
            "and one target column."
        )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Configure MLflow and DagsHub
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("akshaygowda52130")

    with mlflow.start_run():

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        # Perform hyperparameter tuning
        grid_search = hyperparamter_tuning(
            x_train,
            y_train,
            param_grid,
            random_state
        )

        best_model = grid_search.best_estimator_

        # Predict using the trained model
        y_train_pred = best_model.predict(x_train)
        y_pred = best_model.predict(x_test)

        # Generate MLflow signature from model inputs and predictions
        # Convert predictions to a regular Python list to avoid NumPy
        # scalar serialization issues.
        signature = infer_signature(
            x_train,
            y_train_pred.tolist()
        )

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Best Parameters: {grid_search.best_params_}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("random_state", int(random_state))
        mlflow.log_param("test_size", float(test_size))
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", float(accuracy))

        # Log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Log the trained model to DagsHub using MLflow
        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        # Trust only the specific tree type required by this locally
        # trained RandomForest model when using the skops serializer.
        trusted_types = ["sklearn.tree._tree.Tree"]

        log_model_kwargs = {
            "sk_model": best_model,
            "name": "model",
            "signature": signature,
            "skops_trusted_types": trusted_types
        }

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                registered_model_name="best_model",
                **log_model_kwargs
            )
        else:
            mlflow.sklearn.log_model(**log_model_kwargs)

        # Create the model directory if it doesn't exist
        model_dir = os.path.dirname(model_path)

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        # Save the trained model locally
        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)

        print(f"Model saved to {model_path}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print("Training and MLflow logging completed successfully.")


if __name__ == "__main__":

    train_model(
        data_path=params["input"],
        model_path=params["model"],
        random_state=params["random_state"],
        test_size=params["test_size"],
        param_grid=params["param_grid"]
    )
