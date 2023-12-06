import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Masking
from tensorflow.keras.metrics import Precision, AUC
from tensorflow.keras.callbacks import Callback
import os
from matplotlib import pyplot as plt
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from collections import defaultdict
import random
from dataclasses import dataclass
from datetime import datetime
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine


class SaveLastBestModel(Callback):
    def __init__(self, base_save_path, monitor="val_loss", mode="min", verbose=0):
        super(SaveLastBestModel, self).__init__()
        self.base_save_path = base_save_path  # Base path without format specifiers
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.Inf if mode == "min" else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if (self.mode == "min" and current < self.best) or (
            self.mode == "max" and current > self.best
        ):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}")
        elif self.verbose > 0:
            print(
                f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.4f}"
            )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            formatted_save_path = self.base_save_path.format(
                epoch=self.best_epoch + 1, val_loss=self.best
            )
            self.model.save(formatted_save_path)
            if self.verbose > 0:
                print(
                    f"The last best model weights were saved to {formatted_save_path}"
                )


@dataclass
class Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def load_single(filename):
    with open(filename) as f:
        data = json.load(f)
    type = data["type"]
    position = data["position"]
    gyro_data = data["gyro"]
    acc_data = data["acc"]
    x_coords = data["x"]
    y_coords = data["y"]
    acc_properties = data["aproperties"][0]  # Assuming there's only one entry
    gyro_properties = data["gproperties"][0]  # Assuming there's only one entry

    # TODO: So currently it works much better without the gyro and acc than with it lmao, it has to be checked
    combined_data = np.column_stack((x_coords, y_coords))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(combined_data)

    return scaled_data, type, position, filename


def load_data(path):
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                absolute_path = os.path.abspath(os.path.join(root, file))
                json_files.append(absolute_path)

    # List comprehension to read data from files and process each element
    arrays = [load_single(filepath) for filepath in json_files]
    # Splitting the array into 'position' (XX, YY, ZZ) and 'rest' (type, position, filename)
    rest = [(type, pos, fname) for _, type, pos, fname in arrays]
    time_data = [data for data, _, _, _ in arrays]

    time_data = tf.keras.preprocessing.sequence.pad_sequences(
        time_data, padding="post", value=2.0, dtype="float"
    )

    return time_data, rest


def split_data(data):
    train, test, val = [], [], []
    for person, properties in data.items():
        for prop in ["true", "false"]:
            random.shuffle(data[person][prop])
            n = len(properties[prop])
            n_test = max(1, n // 10)  # At least 1 or 10% for test
            n_val = max(1, n // 10)  # At least 1 or 10% for validation

            # Add to test, validation, and training sets
            test.extend(
                [(seq, 1 if prop == "true" else 0) for seq in properties[prop][:n_test]]
            )
            val.extend(
                [
                    (seq, 1 if prop == "true" else 0)
                    for seq in properties[prop][n_test : n_test + n_val]
                ]
            )
            train.extend(
                [
                    (seq, 1 if prop == "true" else 0)
                    for seq in properties[prop][n_test + n_val :]
                ]
            )
    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(val)

    train_data, train_labels = zip(*train)
    test_data, test_labels = zip(*test)
    val_data, val_labels = zip(*val)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    new_data = Data(
        train_data, train_labels, test_data, test_labels, val_data, val_labels
    )

    return new_data


def add_jitter(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def time_warp(data, warp_factor=0.2):
    time_steps = data.shape[1]
    warp_step = int(np.random.uniform(-warp_factor, warp_factor) * time_steps)

    return np.roll(data, shift=warp_step, axis=1)


def random_scaling(data, scaling_factor_range=(0.9, 1.1)):
    factor = np.random.uniform(*scaling_factor_range)
    return data * factor


def augment_sample(sample):
    # Choose an augmentation method
    choice = np.random.choice(["jitter", "scaling", "time_warp"])

    # Apply the chosen augmentation
    if choice == "jitter":
        test = add_jitter(sample)
        if test is None:
            a = 0
        return test
    elif choice == "scaling":
        test = random_scaling(sample)
        if test is None:
            a = 0
        return test
    elif choice == "time_warp":
        test = time_warp(sample)
        if test is None:
            a = 0
        return test


def augmentation(data):
    augmented_x_train = []
    augmented_y_train = []

    def calculate_change_metrics(array1, array2):
        mse = mean_squared_error(array1, array2)
        mae = mean_absolute_error(array1, array2)
        cosine_sim = 1 - cosine(
            array1.flatten(), array2.flatten()
        )  # 1 - cosine distance

        return {"MSE": mse, "MAE": mae, "Cosine Similarity": cosine_sim}

    for sample, label in zip(data.x_train, data.y_train):
        # Keep the original sample and label
        augmented_x_train.append(sample)
        augmented_y_train.append(label)

        # Augment the sample
        augmented_sample = augment_sample(sample)
        metrics = calculate_change_metrics(sample, augmented_sample)
        # Append the augmented sample and its label
        augmented_x_train.append(augmented_sample)
        augmented_y_train.append(label)

    # Convert lists to numpy arrays
    data.x_train = np.array(augmented_x_train)
    data.y_train = np.array(augmented_y_train)


def preprocessing(timedata, metadata):
    data_by_signature_position = defaultdict(list)
    for t_data, meta in zip(timedata, metadata):
        path_parts = os.path.normpath(meta[2]).split(os.sep)
        name_part = path_parts[-1]
        signature = name_part.split("_")[0]
        position = meta[1]
        key = (signature, position)
        data_by_signature_position[key].append(
            (t_data, meta[0])
        )  # (position, Original|Fake)

    person_data = defaultdict(lambda: {"true": [], "false": []})

    # TODO: So this is the option that different types are still the same person
    for (person_id, _), sequences in data_by_signature_position.items():
        for sequence, label in sequences:
            if label == "Original":  # True property
                person_data[person_id]["true"].append(sequence)
            else:  # False property
                person_data[person_id]["false"].append(sequence)

    data = split_data(person_data)

    return data


def train(
    model: tf.keras.Model,
    data: Data,
    callbacks,
    optimizer,
    loss,
    metrics,
    epochs=100,
    batch_size=64,
):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(
        data.x_train,
        data.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data.x_val, data.y_val),
        callbacks=callbacks,
    )

    evaluation_results = model.evaluate(data.x_test, data.y_test)
    test_loss = evaluation_results[0]
    test_metrics = evaluation_results[1:]
    print(f"Test Loss: {test_loss}")
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(f"Test {metric_name}: {metric_value}")

    predictions = model.predict(data.x_test)

    thresholds = np.arange(0.01, 1, 0.01)
    best_threshold = 0
    best_f1 = 0
    f1s = []
    for threshold in thresholds:
        # Convert probabilities to binary labels based on the threshold
        predicted_labels = (predictions > threshold).astype(int)
        f1 = f1_score(data.y_test, predicted_labels)
        f1s.append(f1)

        # Update the best threshold based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    plt.plot(thresholds, f1s)
    plt.show()
    print(f"Best threshold at: {best_threshold}")

    predicted_labels = (predictions > best_threshold).astype(int)
    correct_predictions = np.sum(predicted_labels.flatten() == data.y_test)
    incorrect_predictions = len(data.y_test) - correct_predictions
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")

    # Calculate F1 Score
    f1 = f1_score(data.y_test, predicted_labels, average="binary")
    print(f"F1 Score: {f1}")

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(data.y_test, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate a classification report
    class_report = classification_report(
        data.y_test, predicted_labels, target_names=["Fake", "Original"]
    )
    print("Classification Report:")
    print(class_report)

    labels = ["Fake", "Original"]

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

    return history


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Masking(mask_value=2.0, input_shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32, return_sequences=False),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


if __name__ == "__main__":
    timedata, metadata = load_data("../../data/data/")
    preprocessed_data = preprocessing(timedata, metadata)
    # augmentation(preprocessed_data)
    model = build_lstm_model((None, preprocessed_data.x_train.shape[2]))

    model.summary()

    save_dir = "./lstm/saved/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "weights.{epoch:02d}-{val_loss:.2f}.h5")
    tensorboard_path = os.path.join(
        "tensorboard_lstm", datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        save_freq="epoch",
        save_best_only=True,
        mode="min",
        verbose=1,
        monitor="val_loss",
    )

    save_best_model = SaveLastBestModel(save_path, "val_loss", "min", 1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_path, histogram_freq=1, embeddings_freq=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=30, verbose=1, mode="min"
    )

    history = train(
        model,
        preprocessed_data,
        [save_best_model, tensorboard_callback, early_stopping],
        "adam",
        "binary_crossentropy",
        ["accuracy", Precision(), AUC(curve="PR")],
        300,
        64,
    )
