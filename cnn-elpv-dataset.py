#!/usr/bin/env python3
"""CNN-based identification of defective solar cells in electroluminescence imagery.

@author: Arne Ludwig <arne.ludwig@posteo.de>
@copyright: Copyright © 2022 by Arne Ludwig
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from elpv_dataset.utils.elpv_reader import load_dataset

learning_rate_methods = {"const", "exp-decay", "adaptive"}


# %% Initialization

use_pretrained_type_model = True
learning_rate_method = os.environ.get("LR", "adaptive")

assert (
    learning_rate_method in learning_rate_methods
), f"LR must be one of {', '.join(learning_rate_methods)}"

models_root = Path("./models") / f"{learning_rate_method}-lr"
models_root.mkdir(parents=True, exist_ok=True)

history_path = models_root / "history.json"
metrics_path = models_root / "metrics.json"
model_path = models_root / "model.h5"
type_history_path = models_root / "type_history.json"
type_model_path = models_root / "type_model.h5"

# Only use pre-trained type model if it exists!
if not type_model_path.exists():
    use_pretrained_type_model = False

# Detect whether we are running on a cluster for training only
training_mode = "SLURM_JOB_ID" in os.environ
rng = np.random.default_rng()

if "SLURM_CPUS_PER_TASK" in os.environ:
    # Respect the CPU limitations of the cluster
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    tf.config.threading.set_intra_op_parallelism_threads(cpus)
    tf.config.threading.set_inter_op_parallelism_threads(cpus)

# Avoid importing plotting libraries in training mode because they usually
# takes a couple of seconds to load
if not training_mode:
    import matplotlib.pyplot as plt
    import seaborn as sns


# %% Load data

images, probs, types = load_dataset()


# %% Inspect data

pixel_range = np.amin(images), np.amax(images)
prob_values, probs_counts = np.unique(probs, return_counts=True)
type_values, type_counts = np.unique(types, return_counts=True)

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

    plt.hist(probs)
    plt.tight_layout()
    plt.show()

    plt.hist(types)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=[3.2 * 5, 3.2 * 4])
    samples = images[rng.integers(images.shape[0], size=5 * 4)]
    for i in range(4):
        for j in range(5):
            k = i * 5 + j
            sample = samples[k]
            plt.subplot(4, 5, k + 1)
            plt.imshow(sample)
            plt.axis("image")
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.savefig("solarcell.png", facecolor="black")


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

    return matches.mean(axis=0)


def mean_absolute_error(y_true, y_pred):
    """Compute mean absolute error.

    @returns  Mean absolute errors of shape `y_true.shape[1:]`
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)


def r2_score(y_true, y_pred):
    """Compute coefficient of determination, denoted R².

    From Wikipedia:
        In statistics, the coefficient of determination, denoted R² or r² and
        pronounced "R squared", is the proportion of the variation in the
        dependent variable that is predictable from the independent
        variable(s).

    @returns  Coefficients of determination of shape `y_true.shape[1:]`
    """
    y_diff = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    y_square = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)), axis=0)

    return 1 - y_diff / y_square


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

prob1 = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_1")(
    flat_feature_maps
)
modulated_prob1 = keras.layers.Multiply()([prob1, type_])
prob2 = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_2")(
    flat_feature_maps
)
type_inv = keras.layers.Lambda(lambda x: 1 - x, name="invert_type")(type_)
modulated_prob2 = keras.layers.Multiply()([prob2, type_inv])

combined_prob = keras.layers.Concatenate()([modulated_prob1, modulated_prob2])
prob = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_out")(combined_prob)

model = keras.Model(inputs, [type_, prob], name="elpv")

if learning_rate_method in {"const", "adaptive"}:
    learning_rate = 1e-4
else:
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        1e-4,
        decay_steps=1_000,
        decay_rate=0.99,
        staircase=True,
    )

model.compile(
    loss=["binary_crossentropy", "MSE"],
    loss_weights=[0.5, 1.0],
    optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
    metrics=[["binary_accuracy"], ["MAE", r2_score]],
)
model.summary()


# %% Training

if training_mode:
    training_begin = time.strftime("%Y-%m-%dT%H-%M-%S")

    if use_pretrained_type_model:
        type_model = keras.models.load_model(type_model_path)
    else:
        # 1. Train cell type model only
        type_model.fit(
            X_train,
            y_types_train,
            epochs=15,
            validation_data=(X_test, y_types_test),
            batch_size=8,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_binary_accuracy",
                    min_delta=1e-5,
                    patience=2,
                    verbose=1,
                    restore_best_weights=True,
                ),
            ],
        )
        type_model.save(type_model_path, save_format="h5")
        type_history = dict(type_model.history.history)
        type_history["__timestamp__"] = training_begin
        with type_history_path.open("w") as outfile:
            json.dump(type_history, outfile)

    # Freeze weights of type model
    type_model.trainable = True
    cnn.trainable = True

    # 2. Train defect probability model only

    # Compensate non-uniform distribution of probs
    y_probs_train_values, y_probs_train_counts = np.unique(
        y_probs_train, return_counts=True
    )
    probs_weights = y_probs_train_counts.sum() / y_probs_train_counts
    sample_weights = np.zeros(y_probs_train.shape, dtype=float)
    for prob, weight in zip(y_probs_train_values, probs_weights):
        sample_weights[y_probs_train == prob] = weight

    fit_callbacks = []
    if learning_rate_method in {"adaptive"}:
        fit_callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.8,
                patience=10,
                verbose=1,
                mode="auto",
                min_delta=1e-5,
            )
        )

    model.fit(
        X_train,
        [y_types_train, y_probs_train],
        epochs=300,
        sample_weight=sample_weights,
        validation_data=(X_test, (y_types_test, y_probs_test)),
        batch_size=8,
        callbacks=fit_callbacks,
    )
    model.save(model_path, save_format="h5")
    history = dict(
        (k, [float(v) for v in vs]) for k, vs in model.history.history.items()
    )
    history["__timestamp__"] = training_begin
    with history_path.open("w") as outfile:
        json.dump(history, outfile)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = dict(
        train_type_accuracy=binary_accuracy(y_types_train, train_pred[0]).item(),
        train_prob_mae=mean_absolute_error(y_probs_train, train_pred[1]).item(),
        test_type_accuracy=binary_accuracy(y_types_test, test_pred[0]).item(),
        test_prob_mae=mean_absolute_error(y_probs_test, test_pred[1]).item(),
    )
    metrics["__timestamp__"] = training_begin
    with metrics_path.open("w") as outfile:
        json.dump(metrics, outfile)


# %% Evaluation


def plot_learning_curve(history, name, exclude=[]):
    """Plot learning curve(s) from history dictionary."""
    for key, curve in history.items():
        if any(ex in key for ex in exclude):
            continue

        if key.endswith("loss"):
            ls = "-."
        elif key.startswith("val"):
            ls = "-"
        else:
            ls = ":"
        plt.plot(curve, ls=ls, label=key)
    plt.grid(True)
    plt.legend()
    plt.title(f"Learning Curve of {name}")
    plt.xlabel("Epoch")
    plt.show()


if not training_mode:
    if not use_pretrained_type_model:
        type_model = keras.models.load_model(type_model_path)
        with (type_history_path).open() as infile:
            type_history = json.load(infile)
            print(
                "-- loaded `type_history` for run "
                + type_history.pop("__timestamp__", "N/A")
            )

        plot_learning_curve(type_history, "Type Model")

        test_pred_type = type_model.predict(X_test)

        cm = confusion_matrix(y_types_test, test_pred_type >= 0.5)
        disp = ConfusionMatrixDisplay(cm, display_labels=type_values)
        disp.plot()

    model = keras.models.load_model(model_path, custom_objects=dict(r2_score=r2_score))
    with (history_path).open() as infile:
        history = json.load(infile)
        print("-- loaded `history` for run " + history.pop("__timestamp__", "N/A"))
    with (metrics_path).open() as infile:
        metrics = json.load(infile)
        print("-- loaded `metrics` for run " + metrics.pop("__timestamp__", "N/A"))

    print("Final metrics:")
    for key, val in metrics.items():
        print(f"  {key:<20s}: {val:9.2f}")

    plot_learning_curve(
        history,
        f"Defect Prob. Model\n(LR={learning_rate_method})",
        exclude=["dense_type_1", "MAE"],
    )

    test_pred = model.predict(X_test)

    cm = confusion_matrix(y_types_test, test_pred[0] >= 0.5)
    disp = ConfusionMatrixDisplay(cm, display_labels=type_values)
    disp.plot()
    plt.show()

    plt.figure(figsize=(6.4, 6.4))
    sns.violinplot(y_probs_test.reshape(-1), test_pred[1].reshape(-1), inner="stick")
    plt.title("Defect Probability")
    plt.xticks(range(4), ["0", "⅓", "⅔", "1"])
    plt.xlabel("Test Values")
    plt.ylim([0, 1])
    plt.ylabel("Predicted Values")
    plt.show()

    prob_test_pred_quant = np.round(3 * test_pred[1]) / 3
    plt.figure(figsize=(6.4, 6.4))
    for i, pred_prob in enumerate(prob_values):
        mask_pred = prob_test_pred_quant == pred_prob
        for j, true_prob in enumerate(prob_values):
            mask = (y_probs_test == true_prob) & mask_pred
            mask = mask.reshape(-1)
            sample = X_test[mask][:, :, :, 0]

            row = i
            col = j
            plt.subplot(4, 4, 4 * row + col + 1)

            if len(sample) > 0:
                plt.imshow(sample[0])

            plt.axis("image")
            plt.xticks([])
            plt.yticks([])
            if row == 0:
                plt.title(f"true={true_prob:.1f}")
            if col == 0:
                plt.ylabel(f"pred={pred_prob:.1f}")
    plt.tight_layout()
    plt.show()
