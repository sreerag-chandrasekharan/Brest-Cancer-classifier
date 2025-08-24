import os, json, joblib, yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import mlflow
import mlflow.sklearn

def train(data_dir: str, params_path: str, model_dir: str, metrics_path: str):
    # --- Load processed data
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    # --- Load params
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)["models"]["cytosis_model"]

    # --- Build model
    pipe = Pipeline([
        ('classifier', LogisticRegression(
            C=params['C'],
            solver='liblinear',
            class_weight=params['class_weight'],
            max_iter=params['max_iter'],
            random_state= params['random_state']
        ))
    ])

    # --- Train
    mlflow.set_experiment('breast_cancer_detection_cytosis')
    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        # --- Metrics
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "f1": float(f1_score(y_test, preds)),
            "accuracy": float(accuracy_score(y_test, preds))
        }

        # --- Save model + metrics
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipe, os.path.join(model_dir, "cytosis_model.pkl"))

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # --- Log with MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, "model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/processed_cytosis")
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--modeldir", type=str, default="models/cytosis")
    parser.add_argument("--metrics", type=str, default="metrics_cytosis.json")
    args = parser.parse_args()

    train(args.datadir, args.params, args.modeldir, args.metrics)
