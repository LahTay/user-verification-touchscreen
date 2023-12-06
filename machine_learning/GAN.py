import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from timebudget import timebudget
from multiprocessing import Pool, cpu_count
import machine_learning.preprocessing as preprocessing
import datetime

print(cpu_count())
# allows for computing using tensor cores


tf.keras.mixed_precision.set_global_policy("mixed_float32")


# class WeightNormalizedLayerWrapper(tf.keras.layers.Wrapper):
#     def __init__(self, layer, **kwargs):
#         super(WeightNormalizedLayerWrapper, self).__init__(layer, **kwargs)

#     def build(self, input_shape):
#         super(WeightNormalizedLayerWrapper, self).build(input_shape)

#         # Retrieve the layer's weights after it has been built
#         weights = self.layer.get_weights()

#         if isinstance(self.layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
#             kernel_shape = weights[0].shape
#             self.layer.kernel = self.layer.add_weight("kernel", shape=kernel_shape, initializer="glorot_uniform", trainable=True)
#             output_size = self.compute_conv_output_shape(input_shape)
#         elif isinstance(self.layer, tf.keras.layers.Dense):
#             units = weights[0].shape[-1]
#             self.layer.kernel = self.layer.add_weight("kernel", shape=(input_shape[-1], units), initializer="glorot_uniform", trainable=True)
#             output_size = (units,)
#         else:
#             raise ValueError("Unsupported layer type: {}".format(type(self.layer)))

#         self.layer.g = self.layer.add_weight("g", shape=(self.layer.kernel.shape[-1],), initializer="ones", trainable=True)

#         # Set the output size dynamically based on layer type
#         self.output_size = output_size

#     def call(self, inputs, training=None, **kwargs):
#         if isinstance(self.layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
#             normalized_weights = tf.nn.l2_normalize(self.layer.kernel, axis=(0, 1, 2))
#         elif isinstance(self.layer, tf.keras.layers.Dense):
#             normalized_weights = tf.nn.l2_normalize(self.layer.kernel, axis=0)
#         else:
#             raise ValueError("Unsupported layer type: {}".format(type(self.layer)))

#         scaled_weights = self.layer.g * normalized_weights
#         scaled_weights = tf.cast(scaled_weights, dtype=tf.float32)  # Cast to float32

#         # Update the size of the kernel if it has changed
#         if self.layer.kernel.shape != scaled_weights.shape:
#             self.layer.kernel.assign(scaled_weights)
#             self.layer.g.assign(tf.ones_like(scaled_weights[0]))

#         return super(WeightNormalizedLayerWrapper, self).call(inputs, training=training, **kwargs)

#     def compute_output_shape(self, input_shape):
#         # Explicitly set the output shape based on the dynamically determined output_size
#         return (input_shape[0],) + tuple(self.output_size)

#     def compute_conv_output_shape(self, input_shape):
#         # Compute the output shape for Conv2D and Conv2DTranspose layers
#         dummy_input = tf.ones(shape=(1,) + tuple(input_shape[1:]), dtype=tf.float32)
#         output_size = self.layer.compute_output_shape(dummy_input.shape)

#         return tuple(output_size[1:])  # Exclude batch dimension

#     def get_config(self):
#         config = super().get_config().copy()
#         return config


@timebudget
def load_from_files(size, directory):  # ignores original/false and other parameters
    if directory != "":
        images = []
        files = os.listdir(directory)
        for path in files:
            images.append(preprocessing.generate(size, directory + "/" + path))
        images = tf.cast(images, tf.float32)
        return images
    else:
        return []


@timebudget
def load_from_files_gpu(size, directory):  # ignores original/false and other parameters
    if directory != "":
        json_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    absolute_path = os.path.abspath(os.path.join(root, file))
                    json_files.append(absolute_path)
        (
            images,
            _,
        ) = preprocessing.generategpu(size, json_files)

        print(len(images))
        images = tf.cast(images, tf.float32)
        return images
    else:
        return []


class saver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 and epoch > 1:
            if self.model.mode != 4:
                if self.model.save:
                    save_path = self.model.manager1.save()
            if self.model.mode != 3:
                if self.model.save:
                    save_path = self.model.manager2.save()
            self.model.print_images(epoch)


@timebudget
def run_generations(operation, inputs, pool):
    return pool.map(operation, inputs)


def random_data(num, x):
    processes_pool = Pool(cpu_count())
    inputs = np.ones(num).astype(int) * x
    out = run_generations(preprocessing.generate, inputs, processes_pool)
    return out


def generator_1(x=32, y=32, z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(noise_dim))
    model.add(
        layers.Dense(int(x / 4) * int(y / 4) * 128, use_bias=False, trainable=True)
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(x / 4), int(y / 4), 128)))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 128)

    model.add(
        layers.Conv2DTranspose(
            64 * z, (5, 5), strides=(1, 1), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 4), int(y / 4), 64 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            32 * z, (5, 5), strides=(2, 2), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 2), int(y / 2), 32 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            16 * z, (5, 5), strides=(2, 2), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, x, y, 16 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            z,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    assert model.output_shape == (None, x, y, z)
    model.summary()
    return model


def generator_2(x=128, y=128, z=3):
    model = tf.keras.Sequential()
    model.add(
        layers.InputLayer(
            (
                int(x / 4),
                int(y / 4),
                z,
            )
        )
    )
    model.add(layers.Conv2D(256 * z, (5, 5), strides=(2, 2), padding="same", groups=z))
    model.add(layers.LeakyReLU())

    # model.add(layers.Conv2D(z, (5, 5), strides=(2, 2), padding='same',input_shape=(int(x / 4),int(y / 4), z,),groups=z))
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2D(z, (5, 5), strides=(2, 2), padding='same', input_shape=(int(x / 4), int(y / 4), z,),groups=z))
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(int(x / 8) * int(y / 8) * 64, use_bias=False, trainable=True))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Reshape((int(x / 8), int(y / 8), 64)))
    # assert model.output_shape == (None, int(x / 8), int(y / 8), 64)

    model.add(
        layers.Conv2DTranspose(
            48 * z, (5, 5), strides=(2, 2), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 4), int(y / 4), 48 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            32 * z, (5, 5), strides=(1, 1), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 4), int(y / 4), 32 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            32 * z, (5, 5), strides=(1, 1), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 4), int(y / 4), 32 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            16 * z, (5, 5), strides=(2, 2), padding="same", use_bias=False, groups=z
        )
    )
    assert model.output_shape == (None, int(x / 2), int(y / 2), 16 * z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            z,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    assert model.output_shape == (None, x, y, z)
    model.summary()
    return model


def discriminator_1(x=32, y=32, z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((x, y, z)))
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(1))
    model.summary()
    return model


def discriminator_2(x=128, y=128, z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((x, y, z)))
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(1))
    model.summary()
    return model


class GAN(tf.keras.Model):
    def __init__(self, dims, noise_dim, load, save, mode):
        super().__init__()
        self.x, self.y, self.z = dims
        self.noise_dim = noise_dim
        self.mode = mode
        self.load = load
        self.save = save
        self.d1_loss_tracker = tf.keras.metrics.Mean(name="d1_loss")
        self.g1_loss_tracker = tf.keras.metrics.Mean(name="g1_loss")
        self.d2_loss_tracker = tf.keras.metrics.Mean(name="d2_loss")
        self.g2_loss_tracker = tf.keras.metrics.Mean(name="g2_loss")

        if self.mode != 4:
            self.discriminator1 = discriminator_1(
                int(self.x / 4), int(self.y / 4), self.z
            )
            self.generator1 = generator_1(int(self.x / 4), int(self.y / 4), self.z)
            checkpoint_dir = "/gan/training_checkpoints_1"
            self.checkpoint_prefix_1 = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint1 = tf.train.Checkpoint(
                generator1=self.generator1,
                discriminator1=self.discriminator1,
                step=tf.Variable(1),
            )
            if self.load:
                self.checkpoint1.restore(
                    tf.train.latest_checkpoint(checkpoint_dir)
                ).expect_partial()
            self.manager1 = tf.train.CheckpointManager(
                self.checkpoint1, checkpoint_dir, max_to_keep=3
            )

        if self.mode != 3:
            self.discriminator2 = discriminator_2(self.x, self.y, self.z)
            self.generator2 = generator_2(self.x, self.y, self.z)
            checkpoint_dir = "/gan/training_checkpoints_2"
            self.checkpoint_prefix_2 = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint2 = tf.train.Checkpoint(
                generator2=self.generator2,
                discriminator2=self.discriminator2,
                step=tf.Variable(1),
            )
            if self.load:
                self.checkpoint2.restore(
                    tf.train.latest_checkpoint(checkpoint_dir)
                ).expect_partial()
            self.manager2 = tf.train.CheckpointManager(
                self.checkpoint2, checkpoint_dir, max_to_keep=3
            )

    def compile(
        self, d_optimizer, g_optimizer, loss_fn, d_loss, g_loss, steps_per_execution
    ):
        super().compile(steps_per_execution=steps_per_execution)
        self.d1_optimizer = d_optimizer
        self.g1_optimizer = g_optimizer
        self.d2_optimizer = d_optimizer
        self.g2_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.loss_fn_d = d_loss
        self.loss_fn_g = g_loss

    @tf.function
    def gradient_penalty(self, real_samples, fake_samples, discriminator):
        # # Ensure consistency in data type
        # real_samples = tf.cast(real_samples, tf.float32)
        # fake_samples = tf.cast(fake_samples, tf.float32)

        # Use random alpha values
        alpha = tf.random.uniform(
            shape=[tf.shape(real_samples)[0], 1, 1, 1],
            minval=0.2,
            maxval=0.8,
            dtype=tf.float32,
        )
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        # Check for NaN values in the gradients
        nan_mask = tf.math.is_nan(interpolates)
        any_nan = tf.reduce_any(nan_mask)

        if any_nan:
            interpolates = tf.where(
                nan_mask, tf.zeros_like(interpolates, dtype=tf.float32), interpolates
            )

        # Check for Inf values in the gradients
        inf_mask = tf.math.is_inf(interpolates)
        any_inf = tf.reduce_any(inf_mask)

        if any_inf:
            interpolates = tf.where(
                inf_mask, tf.constant(1, dtype=tf.float32), interpolates
            )

        tf.debugging.check_numerics(
            interpolates, "interpolates check before penalty calculation"
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            d_interpolates = discriminator(interpolates)
        tf.debugging.check_numerics(
            d_interpolates, "d_interpolates check before penalty calculation"
        )
        # Check gradients for NaN or Inf
        gradients = tape.gradient(d_interpolates, interpolates)
        tf.debugging.check_numerics(
            gradients, "Gradients check before penalty calculation"
        )

        # Ensure gradients are not None before calculating slopes
        if gradients is not None:
            # Stabilize the denominator in slopes calculation
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-8)
            gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        else:
            gradient_penalty = 0.0  # or handle the case when gradients are None
        if gradient_penalty < 0:
            gradient_penalty = -gradient_penalty
        return tf.cast(gradient_penalty, tf.float32)

    def train_1(self, real_images, batch_size, random):
        random_latent_vectors = random

        generated_images1 = self.generator1(random_latent_vectors)
        tf.debugging.check_numerics(
            generated_images1,
            "Generated g1 check",
        )
        combined_images = tf.concat(
            [
                generated_images1,
                real_images,
            ],
            axis=0,
        )

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        for _ in range(3):
            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator1(combined_images)
                tf.debugging.check_numerics(
                    predictions,
                    "Predictions d1 check before discriminator loss counting",
                )
                d_loss = self.loss_fn_d(
                    tf.cast(labels, tf.float32), tf.cast(predictions, tf.float32)
                )
                # Compute and add gradient penalty
                gp = self.gradient_penalty(
                    real_images, generated_images1, self.discriminator1
                )
                d_loss += 5 * gp
            grads = tape.gradient(d_loss, self.discriminator1.trainable_weights)
            self.d1_optimizer.apply_gradients(
                zip(grads, self.discriminator1.trainable_weights)
            )
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        misleading_labels = -tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            generated = self.generator1(random_latent_vectors)
            tf.debugging.check_numerics(
                generated, "Generated 1 check before generator loss counting"
            )
            predictions = self.discriminator1(generated)
            tf.debugging.check_numerics(
                predictions, "Predictions d1 check before generator loss counting"
            )
            g_loss = self.loss_fn_g(tf.cast(predictions, tf.float32))
        grads = tape.gradient(g_loss, self.generator1.trainable_weights)
        self.g1_optimizer.apply_gradients(zip(grads, self.generator1.trainable_weights))
        # Update metrics and return their value.
        self.d1_loss_tracker.update_state(d_loss)
        self.g1_loss_tracker.update_state(g_loss)
        return {
            "d1_loss": self.d1_loss_tracker.result(),
            "g1_loss": self.g1_loss_tracker.result(),
            "gp1": gp,
        }

    def train_2(self, real_images, generator_input, batch_size):
        generated_images2 = self.generator2(generator_input)

        combined_images = tf.concat(
            [
                generated_images2,
                real_images,
            ],
            axis=0,
        )

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        for _ in range(3):
            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator2(combined_images)
                tf.debugging.check_numerics(
                    predictions,
                    "Predictions d2 check before discriminator loss counting",
                )
                d_loss = self.loss_fn_d(
                    tf.cast(labels, tf.float32), tf.cast(predictions, tf.float32)
                )
                # Compute and add gradient penalty
                gp = self.gradient_penalty(
                    real_images, generated_images2, self.discriminator2
                )
                d_loss += 5 * gp
            grads = tape.gradient(d_loss, self.discriminator2.trainable_weights)
            self.d2_optimizer.apply_gradients(
                zip(grads, self.discriminator2.trainable_weights)
            )

        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            generated = self.generator2(generator_input)
            tf.debugging.check_numerics(
                generated, "Generated 2 check before generator loss counting"
            )
            predictions = self.discriminator2(generated)
            tf.debugging.check_numerics(
                predictions, "Predictions d2 check before generator loss counting"
            )
            g_loss = self.loss_fn_g(tf.cast(predictions, tf.float32))
        grads = tape.gradient(g_loss, self.generator2.trainable_weights)
        self.g2_optimizer.apply_gradients(zip(grads, self.generator2.trainable_weights))
        # Update metrics and return their value.
        self.d2_loss_tracker.update_state(d_loss)
        self.g2_loss_tracker.update_state(g_loss)
        return {
            "d2_loss": self.d2_loss_tracker.result(),
            "g2_loss": self.g2_loss_tracker.result(),
            "gp2": gp,
        }

    def train_step(self, data):
        real_images, downscaled_images = data
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        real_images = tf.cast(real_images, tf.float32)
        if self.mode == 1:
            random = tf.random.normal(shape=(batch_size, self.noise_dim))
            d1loss = self.train_1(downscaled_images, batch_size, random)
            gen_input = self.generator1(random)
            d2loss = self.train_2(real_images, gen_input, batch_size)
            out = d1loss | d2loss
            return out

        if self.mode == 2:
            random = tf.random.normal(shape=(batch_size, self.noise_dim))
            d1loss = self.train_1(downscaled_images, batch_size, random)
            d2loss = self.train_2(real_images, downscaled_images, batch_size)
            out = d1loss | d2loss
            return out

        if self.mode == 3:
            random = tf.random.normal(shape=(batch_size, self.noise_dim))
            d1loss = self.train_1(downscaled_images, batch_size, random)
            return d1loss

        if self.mode == 4:
            d2loss, g2loss = self.train_2(real_images, downscaled_images, batch_size)
            return d2loss
        # clip_const = 1
        # if self.mode != 4:
        #   for w in self.discriminator1.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -clip_const, clip_const))
        # for w in self.generator1.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -clip_const, clip_const))

        # if self.mode != 3:
        #   for w in self.discriminator2.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -clip_const, clip_const))
        # for w in self.generator2.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -clip_const, clip_const))

    def generate_and_save_images(self, images, epoch, text=""):
        predictions = images

        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            data = tf.cast(predictions[i, :, :, :], tf.float32)
            data = tf.divide(
                tf.subtract(data, tf.reduce_min(data)),
                max(tf.subtract(tf.reduce_max(data), tf.reduce_min(data)), 1e-5),
            )
            plt.imshow(data)
            plt.axis("off")
        plt.savefig(f"/gan/image_at_epoch_{epoch}_{text}.png")
        return predictions

    def print_images(self, epoch):
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        if self.mode == 1:
            self.generate_and_save_images(self.generator1(seed), epoch, "")
            self.generate_and_save_images(
                self.generator2(self.generator1(seed)), epoch, "second_network"
            )
        if self.mode == 2:
            self.generate_and_save_images(self.generator1(seed), epoch, "")
            self.generate_and_save_images(
                self.generator2(self.generator1(seed)), epoch, "second_network"
            )
        if self.mode == 3:
            self.generate_and_save_images(self.generator1(seed), epoch, "")
        if self.mode == 4:
            self.generate_and_save_images(
                self.generator2(self.generator1(seed)), epoch, "second_network"
            )


def discriminator_loss(y_true, y_pred):
    y_sgnm = tf.math.sign(y_pred)
    loss_val = tf.math.sqrt(tf.keras.backend.mean((y_true - y_pred) ** 2))
    loss_sgnm = -tf.keras.backend.mean(y_true * y_sgnm)
    return loss_val + 2 * loss_sgnm


def generator_loss(fake_img):
    y_sgnm = tf.math.sign(fake_img)
    loss_val = tf.keras.backend.mean(fake_img)
    loss_sgnm = tf.keras.backend.mean(y_sgnm)
    return loss_val + 2 * loss_sgnm


if __name__ == "__main__":
    """
    x = rozmiar kwadratu wyjściowego obrazu oraz rozmiar generowanych obrazów przez generator
    num_generated = ilość generowanych obrazów
    num_elements = ilość wylosowanych elementów z wygenerowanych do datasetu (2048 przy 2GB collab 4k i więcej działa)
    epochs = ilość epok uczenia
    noise dim = rozmiar wektora wejściowego do 1 sieci
    num examples to generate = ilosć generowanych obrazów do podglądu uczenia
    save = czy zapisywać checkpointy
    mode = tryb uczenia (1: obie sieci wyjście 1 to wejście 2 (normalna praca) 2: obie sieci uczą się niezależnie 3: tylko 1 sieć 4: tylko 2 sieć
    load = czy załadować ostatni checkpoint
    """
    x = 128
    y = x
    z = 3
    num_generated = 1
    num_elements = 4096
    EPOCHS = 300
    noise_dim = 25
    num_examples_to_generate = 4
    save = True
    mode = 2
    load = False
    images = load_from_files_gpu(x, "../data/data")
    # train_images2 = tf.cast(random_data(num_generated,x),tf.float32)
    # train_images2 = np.vstack((images,train_images2))
    train_images2 = images
    test = list(np.random.choice(train_images2.shape[0], num_elements))
    train_images = []
    for i in test:
        train_images.append(train_images2[i])
    BUFFER_SIZE = num_elements
    BATCH_SIZE = 32
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits

    # generator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
    # discriminator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
    generator_optimizer = tf.keras.optimizers.legacy.RMSprop(1e-4)
    discriminator_optimizer = tf.keras.optimizers.legacy.RMSprop(1e-5)
    train_images = np.array(train_images)
    downscaled_images = tf.cast(
        tf.image.resize(train_images, (int(x / 4), int(y / 4))), tf.float32
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    gan = GAN((x, y, z), noise_dim, load, save, mode)
    gan.compile(
        discriminator_optimizer,
        generator_optimizer,
        cross_entropy,
        discriminator_loss,
        generator_loss,
        steps_per_execution=1,
    )
    gan.fit(
        train_images,
        downscaled_images,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.0,
        max_queue_size=100,
        workers=4,
        use_multiprocessing=True,
        callbacks=[saver(), tensorboard_callback],
        verbose=1,
    )
