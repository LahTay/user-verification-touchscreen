from datetime import datetime

from parser import parse_input
import os
from dataclasses import dataclass
import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from machine_learning.preprocessing import generategpu


if sys.version_info[0:2] != (3, 10):
    raise Exception("It's gonna break if not using python>=3.10 :)")

# If info messages from tf are unwanted use the below statement
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


""" --------------------------------------------------------------------------------------------------------------------
Idea behind this program
It's supposed to be a generic API that would make it extremely easy to teach TUV models
There will be 2 versions of the API: console and straight from the editor
Both versions should be able to do the exact same thing:
    - console using parsed arguments
    - editor changing variable values within the code

There may be some limitations that will make it difficult to make them the same
    - it is possible to write a completely new model in the editor, console would have to use a pre-saved model
    + solution: create a simple script where you can define your model and it will be save which you can later 
    use in the console app
    
    
    
What needs to be implemented:
- Model loading
- Variables setting
- Data loading
- Data preprocessing
- Data augmenting
- Learning process
- Results saving
- Tensorboard implementation
- Console version
""" "------------------------------------------------------------------------------------------------------------------"


@dataclass
class Data:
    """
    Data is stored in following way
    We have a list of users in a given position
    Inside each of those lists there are positions in x and Original|Fake in y
    For example
    x_train[0] -> lista pozycji dla jednego użytkownika o wybranej pozycji
    y_train[0] -> koresponująca lista zawierająca nazwy 'original', 'fake' (mozna zamienic na 0|1)
    """

    x_train: list | np.ndarray
    y_train: list | np.ndarray
    x_test: list | np.ndarray
    y_test: list | np.ndarray

    def convert_to_numpy(self):
        """
        Convenience function changing lists to numpy arrays.
        Lists are easier to create, so it's recommended to finish filling data using lists
        and later change them to numpy arrays.
        The outcome array.shape is:
        (num_people, samples_num, x, y, rgb)
        """
        self.x_train = np.array(
            [np.array(group) for group in self.x_train], dtype=object
        )
        self.y_train = np.array(
            [np.array(group) for group in self.y_train], dtype=object
        )
        self.x_test = np.array([np.array(group) for group in self.x_test], dtype=object)
        self.y_test = np.array([np.array(group) for group in self.y_test], dtype=object)

    def convert_to_ragged_tensor(self):
        """
        This one converts the data into tf tensors.
        It might be better as tf has ragged tensors that allow for different data shapes
        This automatically changes string y values into numbers

        """
        print("Converting data to ragged tensors")

        def convert_single(data):
            data = [tf.convert_to_tensor(group, dtype=tf.float32) for group in data]
            row_lengths = [tensor.shape[0] for tensor in data]
            flattened = tf.concat(data, axis=0)  # Flatten the tensors along axis 0
            data = tf.RaggedTensor.from_row_lengths(flattened, row_lengths)
            return data

        self.change_labels()

        self.x_train = convert_single(self.x_train)
        self.x_test = convert_single(self.x_test)
        self.y_train = convert_single(self.y_train)
        self.y_test = convert_single(self.y_test)

        print("Converting finished")

    def change_labels(self):
        """
        This changes the labels in y arrays from original|fake to 1|0
        """
        for y_group in self.y_train:
            for i, label in enumerate(y_group):
                y_group[i] = 1 if label.lower() == "original" else 0

        for y_group in self.y_test:
            for i, label in enumerate(y_group):
                y_group[i] = 1 if label.lower() == "original" else 0


class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.round(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        false_positives = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
        false_negatives = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + K.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + K.epsilon()
        )
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


def print_gpu():
    if tf.config.list_physical_devices("GPU"):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")


def print_info():
    print_gpu()

    print(tf.__version__)
    print(tf.config.list_physical_devices("GPU"))


def load_data(path, image_size):
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                absolute_path = os.path.abspath(os.path.join(root, file))
                json_files.append(absolute_path)

    position_data, metadata = generategpu(image_size, json_files)

    data_by_signature_position = defaultdict(list)
    for pos_data, meta in zip(position_data, metadata):
        path_parts = os.path.normpath(meta[2]).split(os.sep)
        name_part = path_parts[-1]
        signature = name_part.split("_")[0]
        position = meta[1]
        key = (signature, position)
        data_by_signature_position[key].append(
            (pos_data, meta[0])
        )  # (position, Original|Fake)

    keys = list(data_by_signature_position.keys())
    np.random.shuffle(keys)

    split_index = int(len(keys) * 0.8)
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]

    data = Data([], [], [], [])

    for key in train_keys:
        x_data_group = []
        y_data_group = []
        for pos_data, label in data_by_signature_position[key]:
            x_data_group.append(pos_data)
            y_data_group.append(label)
        data.x_train.append(x_data_group)
        data.y_train.append(y_data_group)

    for key in test_keys:
        x_data_group = []
        y_data_group = []
        for pos_data, label in data_by_signature_position[key]:
            x_data_group.append(pos_data)
            y_data_group.append(label)
        data.x_test.append(x_data_group)
        data.y_test.append(y_data_group)

    return data


def create_base_network(input_shape):
    """
    Base network to be shared (Siamese).
    """
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation="relu")(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu", name="embedding_output")(x)
    return Model(input, x)


def create_model(input_shape):
    """
    Create a Siamese Convolutional model.
    """
    # Create the base network
    base_network = create_base_network(input_shape)
    plot_model(
        base_network,
        to_file="base_network_plot.png",
        show_shapes=True,
        show_layer_names=True,
    )
    # Create the inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Because we re-use the same instance 'base_network',
    # the weights of the network will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Custom layer to compute the absolute difference between the encodings
    # distance = (Lambda(lambda embeddings: tf.math.abs(embeddings[0] - embeddings[1]), name='distance')
    #             ([processed_a, processed_b]))

    distance = Lambda(
        lambda embeddings: tf.sqrt(
            tf.reduce_sum(
                tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True
            )
        ),
        name="distance",
    )([processed_a, processed_b])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation="sigmoid", name="prediction")(distance)

    # Connect the inputs with the outputs
    siamese_model = Model(inputs=[input_a, input_b], outputs=[distance, prediction])

    siamese_model.summary()
    plot_model(
        siamese_model,
        to_file="siamese_model_plot.png",
        show_shapes=True,
        show_layer_names=True,
    )
    return siamese_model


def contrastive_loss(y_true, distance):
    """
    Contrastive loss function.
    :param y_true: If y_true=1 pair is similar, 0 dissimilar.
    :param distance: Shows distance between embeddings.
    :return: Mean of all pairs' losses
    """
    margin = 1  # How far dissimilar labels should be pushed away from each other
    square_pred = tf.square(distance)  # Squaring emphasizes larger errors
    margin_square = tf.square(
        tf.maximum(margin - distance, 0)
    )  # Squared distance between margin and predicted distance
    # Ensure y_true is of the same data type as square_pred and margin_square
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def return_model(model_path, image_size, weights_path=""):
    """
    Return a new model either saved from the path or a new one. Throws exception if wrong path
    :param weights_path: If it's empty no pretraining, else load the weights.
    :param model_path: If it's empty return new model, otherwise return saved model.
    :return: Model
    """
    if model_path == "":
        model = create_model((image_size, image_size, 3))
        if weights_path != "":
            model.load_weights(weights_path)
        return model
    try:
        model = tf.keras.models.load_model(model_path)
        if weights_path != "":
            model.load_weights(weights_path)
        return model
    except Exception as ex:
        print(type(ex))
        print(ex.args)
        print(ex)
        print(
            Rf"Model/weights doesn't exist under path: {model_path} or {weights_path}"
        )


def augmenting(data: Data):
    # TODO: augmenting
    pass


def create_pairs(images, labels):
    """
    So this one creates  pairs of every combination excluding the same pictures and both being fake
    Pair of the 2 pictures in a way: (pic1, pic2), (pic2, pic1) is left as it can help the model learn in both ways
    Idea is that each pair is either (Original, Original) then label is 1 or (Original, Fake) then label is 0
    :param images: Input images
    :param labels: Input labels
    :return: Pairs
    """
    pairs = []
    labels_list = []

    for person_images, person_labels in zip(images, labels):
        num_images = len(person_images)

        # Create pairs of images
        for i in range(num_images):
            for j in range(num_images):
                if i != j:  # Exclude pairs of the same image
                    # Include the pair only if not both images are fake
                    if not (person_labels[i] == 0 and person_labels[j] == 0):
                        pairs.append((person_images[i], person_images[j]))
                        label = (
                            1 if person_labels[i] == 1 and person_labels[j] == 1 else 0
                        )
                        labels_list.append(label)

    return np.array(pairs), np.array(labels_list)


def safe_convert_to_numpy(data):
    try:
        # Attempt to convert to NumPy if it's a TensorFlow tensor
        return data.numpy()
    except AttributeError:
        # If it's already a NumPy array or doesn't have the `.numpy()` method, return it as is
        return data


def model_train(
    model: tf.keras.Model,
    data,
    callbacks,
    optimizer,
    loss,
    metrics,
    generator: tf.keras.preprocessing.image.ImageDataGenerator = None,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    validation_split=0.1,
):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train = create_pairs(data.x_train, data.y_train)
    test = create_pairs(data.x_test, data.y_test)

    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        train[0], train[1], test_size=0.1, random_state=42, stratify=train[1]
    )

    if generator:  # If standard generator is also used not only GAN
        history = model.fit(
            generator.flow(
                [train_pairs[:, 0], train_pairs[:, 1]],
                train_labels,
                batch_size=batch_size,
            ),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            [train_pairs[:, 0], train_pairs[:, 1]],
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
            callbacks=callbacks,
        )

    evaluation_results = model.evaluate(
        [test[0][:, 0], test[0][:, 1]], test[1], verbose=1
    )

    test_loss = evaluation_results[0]
    test_metrics = evaluation_results[1:]
    print(f"Test Loss: {test_loss}")
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(f"Test {metric_name}: {metric_value}")

    predictions = model.predict([test[0][:, 0], test[0][:, 1]])
    predicted_probabilities = predictions[1]

    # Assuming the prediction output is probability of being a genuine pair, threshold it at 0.5
    predicted_labels = (predicted_probabilities > 0.5).astype(int)
    # Since `test[1]` is already a NumPy array, we can use it directly
    true_labels = test[1]
    # Calculate the number of correct predictions
    correct_predictions = np.sum(predicted_labels.flatten() == true_labels)
    incorrect_predictions = len(true_labels) - correct_predictions

    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")

    # Calculate F1 Score
    f1 = f1_score(true_labels, predicted_labels, average="binary")
    print(f"F1 Score: {f1}")

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate a classification report
    class_report = classification_report(
        true_labels, predicted_labels, target_names=["Fake", "Original"]
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


def main():
    print_info()
    parse_input()

    "VARIABLES -------------------------------------------------------------------------------------------------------"
    learning_rate = 0.001
    batch_size = 128

    epoch = 20

    optimizer = tf.keras.optimizers.Adam(  # Here's just some random one, can be from Keras or written by yourself
        learning_rate=learning_rate, beta_1=0.9
    )

    # It's like this because my idea was to have distance as one output.
    # And this distance is learning how far away it should be using contrastive_loss
    # And then from the distance you get prediction whether both are original or one is fake
    loss = {"distance": contrastive_loss, "prediction": "binary_crossentropy"}

    metrics = [F1Score()]

    data_path = "../../data/data"  # Path to input data
    model_path = ""  # If not empty take model from given path
    verbose = True
    save_path = "/saved/weights/weights.{epoch:02d}-{val_loss:.2f}.h5"

    pretrain = ""  # Path to the pretrained weights, empty if learning from scratch

    # This here will work with augemntation, using either GAN or just standard image augmentation
    augment = ""

    tensorboard_path = os.path.join(
        "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path, save_weights_only=True, save_freq="epoch"
    )

    embedding_layer_name = "embedding_output"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_path,
        histogram_freq=1,
        embeddings_freq=1,
        embeddings_layer_names=[embedding_layer_name],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=5, verbose=1, mode="min"
    )

    callbacks = [save_model, tensorboard_callback, early_stopping]

    image_size = 128
    "-----------------------------------------------------------------------------------------------------------------"
    model = return_model(model_path, image_size)
    # Explanation to the structure of the data variable in the class above
    data = load_data(data_path, image_size)
    data.convert_to_ragged_tensor()

    history = model_train(
        model, data, callbacks, optimizer, loss, metrics, None, epoch, batch_size
    )


if __name__ == "__main__":
    main()
