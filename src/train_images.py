
import os
import yaml
import json
import mlflow
import pickle
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

# -------------------------------
# Load params.yaml
# -------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

image_params = params['models']['image_model']

# -------------------------------
# Load datasets from TF format
# -------------------------------
train_ds = tf.data.experimental.load("data/processed/image_data/train")
val_ds = tf.data.experimental.load("data/processed/image_data/val")
test_ds = tf.data.experimental.load("data/processed/image_data/test")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------------------
# Build Base Model
# -------------------------------
base_model = ResNet50(
    weights=image_params["weights"],
    include_top=image_params["include_top"],
    input_shape=tuple(image_params["input_shape"]),
    pooling=image_params["pooling"]
)

# Freeze all layers initially
base_model.trainable = False

# Build classifier head
model = models.Sequential([
    base_model,
    layers.Dense(image_params["dense_units"], activation="relu"),
    layers.Dropout(image_params["dropout"]),
    layers.Dense(1, activation="sigmoid")
])

# -------------------------------
# Compile Initial Model
# -------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=image_params["learning_rate"]),
    loss=image_params["loss"],
    metrics=image_params["metrics"]
)

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_experiment("breast_cancer_images")

with mlflow.start_run():
    mlflow.log_params(image_params)

    # -------------------------------
    # Stage 1: Train Frozen Model
    # -------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=image_params["epochs"],
        batch_size=image_params["batch_size"]
    )


    # -------------------------------
    # Evaluate Model
    # -------------------------------
    results = model.evaluate(test_ds)
    metrics = {name: float(val) for name, val in zip(model.metrics_names, results)}

    # Log metrics
    mlflow.log_metrics(metrics)

    # -------------------------------
    # Save Model
    # -------------------------------
    # Save metrics for DVC
    os.makedirs("results", exist_ok=True)
    with open("results/image_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # save model
    os.makedirs("models", exist_ok=True)

    keras_model_path = "models/image_resnet50.h5"
    pkl_model_path = "models/image_resnet50.pkl"

    model.save(keras_model_path)

    with open(pkl_model_path, "wb") as f:
        pickle.dump({"model_path": keras_model_path}, f)

    # Log to MLflow
    mlflow.log_artifact(keras_model_path)
    mlflow.log_artifact(pkl_model_path)
    mlflow.log_artifact("results/image_metrics.json")

    print(f"Model saved at {keras_model_path} and {pkl_model_path}")
    print("Metrics saved at results/image_metrics.json")