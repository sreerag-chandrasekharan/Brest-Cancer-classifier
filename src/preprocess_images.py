import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Load params
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

img_params = config["dataset"]["images"]
IMG_HEIGHT, IMG_WIDTH = img_params["image_size"]
BATCH_SIZE = config["models"]["image_model"]["batch_size"]
SEED = img_params["seed"]

# Paths
train_dir = img_params["train_path"]
val_dir = img_params["val_path"]
test_dir = img_params["test_path"]
processed_dir = "data/processed/image_data"
os.makedirs(processed_dir, exist_ok=True)

# ImageDataGenerator with rescaling
datagen = ImageDataGenerator(rescale=1./255)

def preprocess_images(directory, shuffle=True):
    dataset = datagen.flow_from_directory(
        directory,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # assuming binary classification
        shuffle=shuffle,
        seed=SEED
    )
    images, labels = [], []
    for _ in range(len(dataset)):
        batch_imgs, batch_labels = next(dataset)
        images.append(batch_imgs)
        labels.append(batch_labels)
    images = tf.convert_to_tensor(tf.concat(images, axis=0), dtype=tf.float32)
    labels = tf.cast(tf.convert_to_tensor(tf.concat(labels, axis=0)), tf.int32)
    labels = tf.cast(labels, tf.int32)
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Process datasets
train_ds = preprocess_images(train_dir, shuffle=img_params["shuffle"])
val_ds = preprocess_images(val_dir, shuffle=False)
test_ds = preprocess_images(test_dir, shuffle=False)

# Save datasets as TF
tf.data.experimental.save(train_ds, os.path.join(processed_dir, "train_ds"))
tf.data.experimental.save(val_ds, os.path.join(processed_dir, "val_ds"))
tf.data.experimental.save(test_ds, os.path.join(processed_dir, "test_ds"))

print("Image preprocessing complete! Saved TF datasets to 'data/processed/image_data'.")
