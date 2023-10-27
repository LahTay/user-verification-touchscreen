
import matplotlib.pyplot as plt
import numpy as np
import os
import TS_option_for_preprocessing
from tensorflow.keras import layers
import tensorflow as tf
from timebudget import timebudget
from multiprocessing import Pool

#allows for computing using tensor cores
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class saver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
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

def random_data(num,x):
    processes_pool = Pool(8)
    inputs = np.ones(num).astype(int)*x
    out = run_generations(TS_option_for_preprocessing.generate, inputs, processes_pool)
    return out

def generator_1(x=32,y=32,z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(noise_dim))
    model.add(layers.Dense(int(x / 4) * int(y / 4) * 128, use_bias=False, trainable=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(x / 4), int(y / 4), 128)))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 128)

    model.add(layers.Conv2DTranspose(64*z, (5, 5), strides=(1, 1), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 64*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32*z, (5, 5), strides=(2, 2), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 2), int(y / 2), 32*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16*z, (5, 5), strides=(2, 2), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, x, y, 16*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(z, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, x, y, z)
    model.summary()
    return model

def generator_2(x=128,y=128,z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((int(x / 4),int(y / 4), z,)))
    model.add(layers.Conv2D(256*z, (5, 5), strides=(2, 2), padding='same',groups=z))
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

    model.add(layers.Conv2DTranspose(48*z, (5, 5), strides=(2, 2), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 48*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32*z, (5, 5), strides=(1, 1), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 32*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32*z, (5, 5), strides=(1, 1), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 4), int(y / 4), 32*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16*z, (5, 5), strides=(2, 2), padding='same', use_bias=False,groups=z))
    assert model.output_shape == (None, int(x / 2), int(y / 2), 16*z)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(z, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, x, y, z)
    model.summary()
    return model

def discriminator_1(x=32,y=32,z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((x,y,z)))
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.Dense(1))
    model.summary()
    return model

def discriminator_2(x=128,y=128,z=3):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((x,y,z)))
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.Dense(1))
    model.summary()
    return model

class GAN(tf.keras.Model):
    def __init__(self, dims, noise_dim,load,save,mode):
        super().__init__()
        self.x,self.y,self.z = dims
        self.noise_dim = noise_dim
        self.mode = mode
        self.load=load
        self.save=save
        self.d1_loss_tracker = tf.keras.metrics.Mean(name="d1_loss")
        self.g1_loss_tracker = tf.keras.metrics.Mean(name="g1_loss")
        self.d2_loss_tracker = tf.keras.metrics.Mean(name="d2_loss")
        self.g2_loss_tracker = tf.keras.metrics.Mean(name="g2_loss")

        if self.mode != 4:
            self.discriminator1 = discriminator_1(int(self.x / 4), int(self.y / 4), self.z)
            self.generator1 = generator_1(int(self.x / 4), int(self.y / 4), self.z)
            checkpoint_dir = './training_checkpoints_1'
            self.checkpoint_prefix_1 = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint1 = tf.train.Checkpoint(generator1=self.generator1,
                                              discriminator1=self.discriminator1,step=tf.Variable(1),)
            if self.load:
                self.checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            self.manager1 = tf.train.CheckpointManager(self.checkpoint1, checkpoint_dir, max_to_keep=3)

        if self.mode != 3:
            self.discriminator2 = discriminator_2(self.x, self.y, self.z)
            self.generator2 = generator_2(self.x, self.y, self.z)
            checkpoint_dir = './training_checkpoints_2'
            self.checkpoint_prefix_2 = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint2 = tf.train.Checkpoint(generator2=self.generator2,
                                              discriminator2=self.discriminator2,step=tf.Variable(1),)
            if self.load:
                self.checkpoint2.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            self.manager2 = tf.train.CheckpointManager(self.checkpoint2, checkpoint_dir, max_to_keep=3)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d1_optimizer = d_optimizer
        self.g1_optimizer = g_optimizer
        self.d2_optimizer = d_optimizer
        self.g2_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_1(self,real_images,batch_size):
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        generated_images1 = self.generator1(random_latent_vectors)
        combined_images = tf.concat([generated_images1, real_images,], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator1(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator1.trainable_weights)
        self.d1_optimizer.apply_gradients(
            zip(grads, self.discriminator1.trainable_weights)
        )
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))

        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator1(self.generator1(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator1.trainable_weights)
        self.g1_optimizer.apply_gradients(zip(grads, self.generator1.trainable_weights))
        # Update metrics and return their value.
        self.d1_loss_tracker.update_state(d_loss)
        self.g1_loss_tracker.update_state(g_loss)
        return {
            "d1_loss": self.d1_loss_tracker.result(),
            "g1_loss": self.g1_loss_tracker.result(),
        }
    def train_2(self,real_images,generator_input,batch_size):
        generated_images2 = self.generator2(generator_input)

        combined_images = tf.concat([generated_images2, real_images,], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator2(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator2.trainable_weights)
        self.d2_optimizer.apply_gradients(
            zip(grads, self.discriminator2.trainable_weights)
        )

        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator2(self.generator2(generator_input))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator2.trainable_weights)
        self.g2_optimizer.apply_gradients(zip(grads, self.generator2.trainable_weights))
        # Update metrics and return their value.
        self.d2_loss_tracker.update_state(d_loss)
        self.g2_loss_tracker.update_state(g_loss)
        return {
            "d2_loss": self.d2_loss_tracker.result(),
            "g2_loss": self.g2_loss_tracker.result(),
        }

    def train_step(self, data):
        real_images,downscaled_images = data
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        real_images=tf.cast(real_images,tf.float16)
        if self.mode == 1:
            d1loss = self.train_1(downscaled_images,batch_size)
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))
            gen_input = self.generator1(random_latent_vectors)
            d2loss = self.train_2(real_images,gen_input,batch_size)
            out = d1loss | d2loss
            return out

        if self.mode == 2:
            d1loss = self.train_1(downscaled_images,batch_size)
            d2loss = self.train_2(real_images,downscaled_images,batch_size)
            out = d1loss | d2loss
            return out

        if self.mode == 3:
            d1loss = self.train_1(downscaled_images,batch_size)
            return d1loss

        if self.mode == 4:
            d2loss,g2loss = self.train_2(real_images,downscaled_images,batch_size)
            return d2loss

    def generate_and_save_images(self,images, epoch, text=""):
        predictions = images

        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(predictions[i, :, :, 0])
            plt.axis('off')
        plt.savefig(f'image_at_epoch_{epoch}_{text}.png')
        return predictions

    def print_images(self,epoch):
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        if self.mode == 1:
            self.generate_and_save_images(self.generator1(seed),epoch,"")
            self.generate_and_save_images(self.generator2(self.generator1(seed)), epoch, "second_network")
        if self.mode == 2:
            self.generate_and_save_images(self.generator1(seed),epoch,"")
            self.generate_and_save_images(self.generator2(self.generator1(seed)), epoch, "")
        if self.mode == 3:
            self.generate_and_save_images(self.generator1(seed),epoch,"")
        if self.mode == 4:
            self.generate_and_save_images(self.generator2(self.generator1(seed)), epoch, "second_network")

if __name__ == '__main__':
    """
    x = rozmiar kwadratu wyjściowego obrazu oraz rozmiar generowanych obrazów przez generator
    num_generated = ilość generowanych obrazów
    num_elements = ilość wylosowanych elementów z wygenerowanych do datasetu
    epochs = ilość epok uczenia
    noise dim = rozmiar wektora wejściowego do 1 sieci
    num examples to generate = ilosć generowanych obrazów do podglądu uczenia
    save = czy zapisywać checkpointy
    mode = tryb uczenia (1: obie sieci wyjście 1 to wejście 2 (normalna praca) 2: obie sieci uczą się niezależnie 3: tylko 1 sieć 4: tylko 2 sieć
    load = czy załadować ostatni checkpoint
    """
    x=128
    y=x
    z=3
    num_generated = 64
    num_elements = 2048
    EPOCHS = 50
    noise_dim = 25
    num_examples_to_generate = 4
    save = True
    mode = 2
    load = False
    train_images2 = random_data(num_generated,x)
    test = list(np.random.choice(num_generated,num_elements))
    train_images = []
    for i in test:
        train_images.append(train_images2[i])
    BUFFER_SIZE = num_elements
    BATCH_SIZE = 32

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    train_images = np.array(train_images)
    downscaled_images = tf.cast(tf.image.resize(train_images, (int(x / 4), int(y / 4))), tf.float16)
    gan = GAN((x,y,z),noise_dim,load,save,mode)
    gan.compile(discriminator_optimizer,generator_optimizer,cross_entropy)
    gan.fit(train_images,downscaled_images,batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.0,
    max_queue_size=100,
    workers=4,
    use_multiprocessing=True,
    callbacks=[saver()]

)
