import tensorflow as tf
import yaml
import os

# Load config
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

IMG_HEIGHT, IMG_WIDTH = config["dataset"]["images"]["image_size"]
BATCH_SIZE = config["models"]["image_model"]["batch_size"]

train_dir = config["dataset"]["images"]["train_path"]
val_dir = config["dataset"]["images"]["val_path"]
test_dir = config["dataset"]["images"]["test_path"]

save_dir = "data/processed/image_data"
os.makedirs(save_dir, exist_ok=True)

def load_and_preprocess(path, batch_size=BATCH_SIZE):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode="int",
        color_mode="rgb",
        shuffle=True
    )
    # Normalize images
    ds = ds.map(lambda x, y: (x / 255.0, y))
    return ds

# Preprocess datasets
train_ds = load_and_preprocess(train_dir)
val_ds = load_and_preprocess(val_dir)
test_ds = load_and_preprocess(test_dir)

print("Preprocessing complete. Datasets saved in TF format at", save_dir)
