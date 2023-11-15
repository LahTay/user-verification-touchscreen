from datetime import datetime

from parser import parse_input
import os
from dataclasses import dataclass
import sys
import numpy as np
import tensorflow as tf



from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical



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
""" '------------------------------------------------------------------------------------------------------------------'


@dataclass
class Data:
    x_train: list | np.ndarray
    y_train: list | np.ndarray
    x_test: list | np.ndarray
    y_test: list | np.ndarray

    def convert_to_numpy(self):
        """
        Convenience function changing lists to numpy arrays.
        Lists are easier to create, so it's recommended to finish filling data using lists
        and later change them to numpy arrays.
        """
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)


def print_gpu():
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")


def print_info():
    print_gpu()

    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))


def load_data(path):
    # TODO: Some way of loading data

    data = Data([], [], [], [])
    return data


def augmenting(data:Data):
    # TODO: augmenting
    pass


def preprocessing(data: Data):
    # TODO: preprocessing


    processed_data = []

    return processed_data





def model_train(model: tf.keras.Model, data, callbacks, optimizer, loss, metrics,
                generator: tf.keras.preprocessing.image.ImageDataGenerator =None, epochs=100, batch_size=64,
                learning_rate=0.001, validation_split=0.1):

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    if generator:  # If standard generator is also used not only GAN
        history = model.fit(generator.flow(data.x_train, data.y_train, batch_size=batch_size),
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            callbacks=callbacks)
    else:
        history = model.fit(data.x_train,
                            data.y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            callbacks=callbacks,
                            )

    return history


def create_model():
    """
    Create a new model in here if it's not preloaded.
    :return: Model
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def return_model(model_path, weights_path=''):
    """
    Return a new model either saved from the path or a new one. Throws exception if wrong path
    :param weights_path: If it's empty no pretraining, else load the weights.
    :param model_path: If it's empty return new model, otherwise return saved model.
    :return: Model
    """
    if model_path == '':
        model = create_model()
        if weights_path != '':
            model.load_weights(weights_path)
        return model
    try:
        model = tf.keras.models.load_model(model_path)
        if weights_path != '':
            model.load_weights(weights_path)
        return model
    except Exception as ex:
        print(type(ex))
        print(ex.args)
        print(ex)
        print(Rf"Model/weights doesn't exist under path: {model_path} or {weights_path}")


def main():
    print_info()
    parse_input()

    'VARIABLES -------------------------------------------------------------------------------------------------------'
    learning_rate = 0.001
    batch_size = 128

    epoch = 100

    optimizer = tf.keras.optimizers.SGD(  # Here's just some random one, can be from Keras or written by yourself
        learning_rate=learning_rate, momentum=0.9, nesterov=True)

    loss = 'categorical_crossentropy'

    metrics = ['accuracy', 'recall']




    data_path = ''  # Path to input data
    model_path = ''  # If not empty take model from given path
    verbose = True
    save_path = '/saved/weights/'

    pretrain = ''  # Path to the pretrained weights, empty if learning from scratch


    # This here will work with augemntation, using either GAN or just standard image augmentation
    augment = ''


    tensorboard_path = os.path.join("tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))

    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                    save_weights_only=True,
                                                    save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_path, histogram_freq=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                      patience=10, verbose=1, mode='min')

    callbacks = [save_model, tensorboard_callback, early_stopping]

    '-----------------------------------------------------------------------------------------------------------------'
    model = return_model(model_path)
    data = load_data(data_path)

    processed_data = preprocessing(data)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    processed_data = Data(train_images, train_labels, test_images, test_labels)

    print("x_train shape:", processed_data.x_train.shape)
    print("y_train shape:", processed_data.y_train.shape)


    history = model_train(model, processed_data, callbacks, 'adam', loss, ['accuracy'],
                          None, 50, batch_size)





if __name__ == '__main__':
    main()









