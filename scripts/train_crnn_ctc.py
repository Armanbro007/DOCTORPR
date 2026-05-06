from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_DIR, MODEL_PATH, VOCAB_PATH
from src.dataset import load_labels
from src.preprocessing import preprocess_path

BATCH_SIZE = 32
EPOCHS = 30


def build_vocab(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    characters = sorted(set("".join(labels)))
    char_to_idx = {char: index for index, char in enumerate(characters)}
    return characters, char_to_idx


def encode_label(text: str, char_to_idx: dict[str, int], max_len: int) -> np.ndarray:
    encoded = [char_to_idx[char] for char in text]
    padded = encoded + [-1] * (max_len - len(encoded))
    return np.array(padded, dtype=np.int32)


def dataframe_to_arrays(df, char_to_idx: dict[str, int], max_label_len: int):
    images = []
    labels = []
    label_lengths = []
    valid_rows = []

    for row in df.itertuples(index=False):
        image_path = Path(row.image_path)
        label = str(row.medicine_name)
        if not image_path.exists() or not label:
            continue
        images.append(preprocess_path(image_path).T[..., np.newaxis])
        labels.append(encode_label(label, char_to_idx, max_label_len))
        label_lengths.append(len(label))
        valid_rows.append(row)

    if not images:
        raise ValueError("No trainable images were found.")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(label_lengths, dtype=np.int32),
        valid_rows,
    )


def build_crnn(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image")

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)

    new_shape = (x.shape[1], x.shape[2] * x.shape[3])
    x = tf.keras.layers.Reshape(target_shape=new_shape)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    outputs = tf.keras.layers.Dense(num_classes + 1, activation="softmax", name="character_probs")(x)
    return tf.keras.Model(inputs, outputs, name="crnn_ctc")


def ctc_loss(labels, predictions, label_lengths):
    batch_size = tf.shape(predictions)[0]
    time_steps = tf.shape(predictions)[1]
    input_lengths = tf.fill([batch_size, 1], time_steps)
    label_lengths = tf.expand_dims(label_lengths, axis=1)
    return tf.keras.backend.ctc_batch_cost(labels, predictions, input_lengths, label_lengths)


class CTCLossModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model):
        super().__init__()
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        images, labels, label_lengths = data
        with tf.GradientTape() as tape:
            predictions = self.base_model(images, training=True)
            loss = tf.reduce_mean(ctc_loss(labels, predictions, label_lengths))
        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        images, labels, label_lengths = data
        predictions = self.base_model(images, training=False)
        loss = tf.reduce_mean(ctc_loss(labels, predictions, label_lengths))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def make_dataset(images, labels, label_lengths, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels, label_lengths))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(images), 2000), seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    train_df = load_labels("train")
    valid_df = load_labels("validation")
    train_df = train_df.dropna(subset=["medicine_name"])
    valid_df = valid_df.dropna(subset=["medicine_name"])

    characters, char_to_idx = build_vocab(train_df["medicine_name"].astype(str).tolist())
    max_label_len = int(train_df["medicine_name"].astype(str).str.len().max())

    x_train, y_train, train_label_lengths, _ = dataframe_to_arrays(train_df, char_to_idx, max_label_len)
    x_valid, y_valid, valid_label_lengths, _ = dataframe_to_arrays(valid_df, char_to_idx, max_label_len)

    base_model = build_crnn(num_classes=len(characters))
    train_model = CTCLossModel(base_model)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    train_model.fit(
        make_dataset(x_train, y_train, train_label_lengths, shuffle=True),
        validation_data=make_dataset(x_valid, y_valid, valid_label_lengths, shuffle=False),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    base_model.save(MODEL_PATH)
    VOCAB_PATH.write_text(
        json.dumps(
            {
                "characters": characters,
                "char_to_idx": char_to_idx,
                "idx_to_char": {index: char for char, index in char_to_idx.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")


if __name__ == "__main__":
    main()
