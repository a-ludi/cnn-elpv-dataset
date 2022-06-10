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

architectures = {"v1", "v2", "v3"}
learning_rate_methods = {"const", "exp-decay", "adaptive"}


# %% Initialization

use_pretrained_type_model = True
architecture = os.environ.get("ARCH", "v3")
learning_rate_method = os.environ.get("LR", "adaptive")

assert (
    learning_rate_method in learning_rate_methods
), f"LR must be one of {', '.join(learning_rate_methods)}"

models_root = Path("./models") / architecture / f"{learning_rate_method}-lr"
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

# Number of data points
N = images.shape[0]
pixel_range = np.amin(images), np.amax(images)
prob_values, probs_counts = np.unique(probs, return_counts=True)
type_values, type_counts = np.unique(types, return_counts=True)

if not training_mode:
    print("Dataset Characteristics")
    print("-----------------------")
    print("Input:")
    print(f"  {N} images of size {'x'.join(str(d) for d in images.shape[1:])}")
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
    plt.savefig("solarcell_samples.png")
    plt.show()

    plt.hist(probs)
    plt.title("Defect Probabilities")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("dist_probs.png")
    plt.show()

    plt.hist(types)
    plt.title("Cell types")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("dist_types.png")
    plt.show()

    samples_grid = (4, 5)
    plt.figure(figsize=[3.2 * samples_grid[1], 3.2 * samples_grid[0]])
    samples = images[rng.integers(N, size=samples_grid[0] * samples_grid[1])]
    for i in range(samples_grid[0]):
        for j in range(samples_grid[1]):
            k = i * 5 + j
            sample = samples[k]
            plt.subplot(samples_grid[0], samples_grid[1], k + 1)
            plt.imshow(sample)
            plt.axis("image")
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.savefig("solarcell.png", facecolor="black")
    plt.show()


# %% Data Preprocessing

# This is used to deterministically choose a random set of data points
# for final validation.
data_rng = np.random.default_rng(1899)

validation_set = np.zeros((N,), dtype=bool)
validation_set[data_rng.integers(N, size=int(N * 0.2))] = True

if training_mode:
    # Remove final validation data before preprocessing
    pre_selector = ~validation_set
else:
    # Reduce to the validation set before preprocessing
    pre_selector = validation_set

images = images[pre_selector]
probs = probs[pre_selector]
types = types[pre_selector]

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
    # Provide proper names for validation data
    X_val = X
    y_types_val = y_types
    y_probs_val = y_probs


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


class NormalNoise(keras.layers.Layer):
    def __init__(self, *args, stddev=0.05, training=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = training
        self.stddev = stddev

    def get_config(self):
        return dict(
            training=self.training,
            stddev=self.stddev,
        )

    def call(self, x):
        if self.training:
            noise = tf.random.normal(
                x.shape[:-1],
                stddev=self.stddev,
                dtype=x.dtype,
            )
            for i in range(x.shape[-1]):
                x[:, :, :, i] += noise
        return x


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

data_augmentation = keras.models.Sequential()
data_augmentation.add(keras.layers.RandomFlip())
data_augmentation.add(NormalNoise(stddev=0.05))

aug_inputs = data_augmentation(inputs)
feature_maps = cnn(aug_inputs)
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


class BinaryMultiplexer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x):
        mask = tf.round(x[0])
        return mask * x[1] + (1 - mask) * x[2]


if architecture == "v1":
    features_type = keras.layers.Concatenate()([flat_feature_maps, type_])
    prob = keras.layers.Dense(1024, activation="sigmoid", name="dense_prob_1")(
        features_type
    )
    prob = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_2")(prob)
elif architecture in {"v2", "v3"}:
    prob1 = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_1")(
        flat_feature_maps
    )

    modulated_prob1 = keras.layers.Multiply()([prob1, type_])
    prob2 = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_2")(
        flat_feature_maps
    )

    if architecture == "v2":
        type_inv = keras.layers.Lambda(lambda x: 1 - x, name="invert_type")(type_)
        modulated_prob2 = keras.layers.Multiply()([prob2, type_inv])

        combined_prob = keras.layers.Concatenate()([modulated_prob1, modulated_prob2])
        prob = keras.layers.Dense(1, activation="sigmoid", name="dense_prob_out")(
            combined_prob
        )
    elif architecture == "v3":
        prob = BinaryMultiplexer(name="switch_prob")([type_, prob1, prob2])

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
        type_model = keras.models.load_model(
            type_model_path,
            custom_objects=dict(
                NormalNoise=NormalNoise,
            ),
        )
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
        epochs=500,
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
        train_prob_r2_score=r2_score(y_probs_train, train_pred[1]).numpy().item(),
        test_type_accuracy=binary_accuracy(y_types_test, test_pred[0]).item(),
        test_prob_mae=mean_absolute_error(y_probs_test, test_pred[1]).item(),
        test_prob_r2_score=r2_score(y_probs_test, test_pred[1]).numpy().item(),
    )
    metrics["__timestamp__"] = training_begin
    with metrics_path.open("w") as outfile:
        json.dump(metrics, outfile)


# %% Evaluation


def plot_learning_curve(history, name, exclude=[], save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


if not training_mode:
    if not use_pretrained_type_model:
        type_model = keras.models.load_model(
            type_model_path,
            custom_objects=dict(
                NormalNoise=NormalNoise,
            ),
        )
        with (type_history_path).open() as infile:
            type_history = json.load(infile)
            print(
                "-- loaded `type_history` for run "
                + type_history.pop("__timestamp__", "N/A")
            )

        plot_learning_curve(type_history, "Type Model")

        val_pred_type = type_model.predict(X_val)

        cm = confusion_matrix(y_types_val, val_pred_type >= 0.5)
        disp = ConfusionMatrixDisplay(cm, display_labels=type_values)
        disp.plot()

    model = keras.models.load_model(
        model_path,
        custom_objects=dict(
            r2_score=r2_score,
            NormalNoise=NormalNoise,
            BinaryMultiplexer=BinaryMultiplexer,
        ),
    )
    with (history_path).open() as infile:
        history = json.load(infile)
        print("-- loaded `history` for run " + history.pop("__timestamp__", "N/A"))
    with (metrics_path).open() as infile:
        metrics = json.load(infile)
        print("-- loaded `metrics` for run " + metrics.pop("__timestamp__", "N/A"))

    print("Final metrics:")
    for key, val in metrics.items():
        print(f"  {key:<20s}: {val:9.2f}")

    if "lr" in history:
        history_scaled = dict(history)
        scaled_lr = 1e4 * np.array(history_scaled.pop("lr"))
        history_scaled["-1e4_learning_rate"] = scaled_lr
        plot_learning_curve(
            history_scaled,
            f"Defect Prob. Model\n(LR={learning_rate_method}, ARCH={architecture})",
            exclude=["dense_type_1", "MAE"],
            save_path="learning_curve.png",
        )
        del history_scaled
    else:
        plot_learning_curve(
            history,
            f"Defect Prob. Model\n(LR={learning_rate_method}, ARCH={architecture})",
            exclude=["dense_type_1", "MAE"],
        )

    val_pred = model.predict(X_val)

    cm = confusion_matrix(y_types_val, val_pred[0] >= 0.5)
    disp = ConfusionMatrixDisplay(cm, display_labels=type_values)
    disp.plot()
    plt.savefig("types_confusion_matrix.png", dpi=200)
    plt.show()

    plt.figure(figsize=(6.4, 6.4))
    sns.violinplot(x=y_probs_val.reshape(-1), y=val_pred[1].reshape(-1), inner="stick")
    plt.title("Defect Probability")
    plt.xticks(range(4), ["0", "⅓", "⅔", "1"])
    plt.xlabel("Validation Values")
    plt.ylim([0, 1])
    plt.ylabel("Predicted Values")
    plt.savefig("prob_regression.png", dpi=200)
    plt.show()

    prob_val_pred_quant = np.round(3 * val_pred[1]) / 3
    plt.figure(figsize=(6.4, 6.4))
    for i, pred_prob in enumerate(prob_values):
        mask_pred = prob_val_pred_quant == pred_prob
        for j, true_prob in enumerate(prob_values):
            mask = (y_probs_val == true_prob) & mask_pred
            mask = mask.reshape(-1)
            sample = X_val[mask][:, :, :, 0]

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

    # %% Quiz

    n_quiz = 8
    quiz_sample = rng.integers(X_val.shape[0], size=n_quiz)
    for i, k in enumerate(quiz_sample):
        plt.figure(figsize=(3.2, 3.2), facecolor="black")
        sample = X_val[k, :, :, 0]
        type_ = type_values[int(val_pred[0][k].round())]
        prob = val_pred[1][k].item()

        plt.imshow(sample)
        plt.axis("image")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"Type: {type_}     Defect Prob.: {prob:.2f}", color="white")
        plt.tight_layout()
        plt.savefig(f"quiz-{i}.png", dpi=200)
        plt.show()
