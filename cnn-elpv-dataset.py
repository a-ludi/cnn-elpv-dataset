#!/usr/bin/env python3
"""CNN-based identification of defective solar cells in electroluminescence imagery.

@author: Arne Ludwig <arne.ludwig@posteo.de>
@copyright: Copyright © 2022 by Arne Ludwig
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from elpv_dataset.utils.elpv_reader import load_dataset

# %% Initialization

models_root = Path("./models")
models_root.mkdir(parents=True, exist_ok=True)

# Detect whether we are running on a cluster for training only
training_mode = "SLURM_JOB_ID" in os.environ
rng = np.random.default_rng()

if "SLURM_CPUS_PER_TASK" in os.environ:
    # Respect the CPU limitations of the cluster
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    tf.config.threading.set_intra_op_parallelism_threads(cpus)
    tf.config.threading.set_inter_op_parallelism_threads(cpus)


# %% Load data

images, probs, types = load_dataset()


# %% Inspect data

pixel_range = np.amin(images), np.amax(images)
prob_values = np.unique(probs)
type_values = np.unique(types)

if not training_mode:
    print("Dataset Characteristics")
    print("-----------------------")
    print("Input:")
    print(
        f"  {images.shape[0]} images of size "
        f"{'x'.join(str(d) for d in images.shape[1:])}"
    )
    print(
        f"  Pixels are in range {pixel_range[0]}-{pixel_range[1]} "
        f"of type {images.dtype}"
    )
    print("Labels:")
    print(
        f"- defect probabilities ({', '.join('{:.2f}'.format(p) for p in prob_values)})"
    )
    print(f"- PV cell types ({', '.join(type_values)})")
    print()

    n_samples = 3
    plt.figure(figsize=(6.4, n_samples * 3.2))
    k = 0
    for i, type_ in enumerate(["mono"] * n_samples + ["poly"] * n_samples):
        for j in range(4):
            prob = j / 3
            sample = images[(probs == prob) & (types == type_)][i % n_samples]
            k += 1
            plt.subplot(2 * n_samples, 4, k)
            plt.imshow(sample)
            plt.axis("image")
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.title(f"p = {j}/3")
            if j == 0 and i % n_samples == 0:
                plt.ylabel(f"← {type_}")
    plt.tight_layout()
    plt.show()


# %% Data Preprocessing

if not training_mode:
    # Reduce memory usage by selecting a "test set" before preprocessing
    sample = rng.integers(images.shape[0], size=500)
    images[sample] = images
    probs[sample] = probs
    types[sample] = types

# Duplicate gray-channel to get RGB images
X = np.stack([images] * 3, axis=-1)
# Normalize images to interval [-1, 1]
X = X.astype(float) / (255 * 2) - 1

# No conversion needed as the values are already in [0, 1] and
# we are modelling the problem as a regression.
y_probs = probs.reshape(-1, 1)

# One-hot encoding of the types
y_types = np.zeros((types.shape[0], 1), dtype=bool)
y_types[types == type_values[1]] = 1

if training_mode:
    # Split in training and test data
    (
        X_train,
        X_test,
        y_types_train,
        y_types_test,
        y_probs_train,
        y_probs_test,
    ) = train_test_split(X, y_types, y_probs, test_size=0.2)
else:
    # Fake splitting in training and test data:
    # - use NO DATA for training
    # - use pre-selected random sample for test
    X_train = np.empty_like(X)
    y_types_train = np.empty_like(y_types)
    y_probs_train = np.empty_like(y_probs)
    X_test = X
    y_types_test = y_types
    y_probs_test = y_probs


# %% Model definition


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """Compute binary accuracy.

    @returns  Binary accuracy values of shape `y_true.shape[1:]`
    """
    matches = y_true.astype(bool) == (y_pred >= threshold)

    return matches.sum(axis=0) / matches.shape[0]


def mean_absolute_error(y_true, y_pred):
    """Compute mean absolute error.

    @returns  Mean absolute errors of shape `y_true.shape[1:]`
    """
    return np.sum(np.abs(y_true - y_pred), axis=0)


cnn = keras.applications.MobileNetV3Large(
    input_shape=X.shape[1:],
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    pooling=None,
    include_preprocessing=False,
)
# Do not train CNN for now
cnn.trainable = False

inputs = keras.layers.Input(X.shape[1:])
feature_maps = cnn(inputs)
flat_feature_maps = keras.layers.Flatten()(feature_maps)
type_ = keras.layers.Dense(1, activation="sigmoid", name="dense_type_1")(
    flat_feature_maps
)
type_model = keras.Model(inputs, type_, name="cell_type")
type_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
    metrics="binary_accuracy",
)

features_type = keras.layers.Concatenate()([flat_feature_maps, type_])
prob = keras.layers.Dense(1024, activation="sigmoid", name="dense_prob_1")(
    features_type
)
prob = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_2")(prob)
model = keras.Model(inputs, [type_, prob], name="elpv")


model.compile(
    loss=["binary_crossentropy", "MSE"],
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-2),
    metrics=[["binary_accuracy"], ["MAE"]],
)
model.summary()


# %% Training

if training_mode:
    # 1. Train cell type model only
    type_model.fit(
        X_train,
        y_types_train,
        epochs=15,
        validation_data=(X_test, y_types_test),
        batch_size=8,
    )
    type_model.save(models_root / "type_model.h5", save_format="h5")
    type_history = dict(type_model.history.history)
    with (models_root / "type_history.json").open("w") as outfile:
        json.dump(type_history, outfile)

    # Freeze weights of type model
    type_model.trainable = False

    # 2. Train defect probability model only
    model.fit(
        X_train,
        [y_types_train, y_probs_train],
        epochs=15,
        validation_data=(X_test, (y_types_test, y_probs_test)),
        batch_size=8,
    )
    model.save(models_root / "model.h5", save_format="h5")
    history = dict(model.history.history)
    with (models_root / "history.json").open("w") as outfile:
        json.dump(history, outfile)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = dict(
        train_type_accuracy=binary_accuracy(y_types_train, train_pred[0]).item(),
        train_prob_mae=mean_absolute_error(y_probs_train, train_pred[1]).item(),
        test_type_accuracy=binary_accuracy(y_types_test, test_pred[0]).item(),
        test_prob_mae=mean_absolute_error(y_probs_test, test_pred[1]).item(),
    )
    with (models_root / "metrics.json").open("w") as outfile:
        json.dump(metrics, outfile)


# %% Evaluation

if not training_mode:
    model = keras.models.load_model(models_root / "model.h5")
    with (models_root / "type_history.json").open() as infile:
        type_history = json.load(infile)
    with (models_root / "history.json").open() as infile:
        history = json.load(infile)
    with (models_root / "metrics.json").open() as infile:
        metrics = json.load(infile)

    print("Final metrics:")
    for key, val in metrics.items():
        print(f"  {key:<20s}: {val:9.2f}")

    for key, hist in type_history.items():
        if key.startswith("val"):
            if key.endswith("loss"):
                ls = "--"
            else:
                ls = "-"
        else:
            if key.endswith("loss"):
                ls = "-."
            else:
                ls = ":"
        plt.plot(hist, ls=ls, label=key)
    plt.grid(True)
    plt.legend(loc=1)
    plt.title("Learning Curve of Type Model")
    plt.xlabel("Epoch")
    plt.show()

    for key, hist in history.items():
        if "dense_type_1" in key:
            continue

        if key.startswith("val"):
            if key.endswith("loss"):
                ls = "--"
            else:
                ls = "-"
        else:
            if key.endswith("loss"):
                ls = "-."
            else:
                ls = ":"
        plt.plot(hist, ls=ls, label=key)
    plt.grid(True)
    plt.legend(loc=1)
    plt.title("Learning Curve of Full Model")
    plt.xlabel("Epoch")
    plt.show()
