
import matplotlib.pyplot as plt
import numpy as np
import os
import TS_option_for_preprocessing
from tensorflow.keras import layers
import time
import tensorflow as tf
from IPython import display
from timebudget import timebudget
from multiprocessing import Pool

#allows for computing using tensor cores
tf.keras.mixed_precision.set_global_policy('mixed_float16')

@timebudget
def run_generations(operation, inputs, pool):
    return pool.map(operation, inputs)

def random_data(num,x):
    processes_pool = Pool(6)
    inputs = np.ones(num).astype(int)*x
    out = run_generations(TS_option_for_preprocessing.generate, inputs, processes_pool)
    return out


def generator_1(x=32,y=32,z=3):
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(x / 4) * int(y / 4) * 128, use_bias=False, input_shape=(noise_dim,), trainable=True))
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

    model.add(layers.Conv2D(256*z, (5, 5), strides=(2, 2), padding='same',input_shape=(int(x / 4),int(y / 4), z,),groups=z))
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

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same',input_shape=[x,y,z]))
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

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same',input_shape=[x,y,z]))
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

def make_generator_model(x=100,y=100,z=3):
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(x/4)*int(y/4)*128, use_bias=False, input_shape=(noise_dim,),trainable=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(x/4), int(y/4), 128)))
    assert model.output_shape == (None, int(x/4), int(y/4), 128)

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(x/4), int(y/4), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(x/2), int(y/2), 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(z, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, x, y, z)
    model.summary()
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[100, 100, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.Dense(1))
    model.summary()
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step_both(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator1(noise, training=True)

      real_output = discriminator1(tf.image.resize(images,[32,32]), training=True)
      fake_output = discriminator1(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator1.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator1.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator1.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator1.trainable_variables))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator2(generated_images, training=True)

      real_output = discriminator2(images, training=True)
      fake_output = discriminator2(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator2.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator2.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator2.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator2.trainable_variables))

@tf.function
def train_step_both_separate(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator1(noise, training=True)

      real_output = discriminator1(tf.image.resize(images,[32,32]), training=True)
      fake_output = discriminator1(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator1.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator1.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator1.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator1.trainable_variables))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator2(tf.image.resize(images,[32,32]), training=True)

      real_output = discriminator2(images, training=True)
      fake_output = discriminator2(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator2.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator2.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator2.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator2.trainable_variables))

@tf.function
def train_step_only1(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator1(noise, training=True)

      real_output = discriminator1(tf.image.resize(images,[32,32]), training=True)
      fake_output = discriminator1(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator1.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator1.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator1.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator1.trainable_variables))

@tf.function
def train_step_only2(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator2(tf.image.resize(images,[32,32]), training=True)

      real_output = discriminator2(images, training=True)
      fake_output = discriminator2(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator2.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator2.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator2.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator2.trainable_variables))

def train(dataset, epochs,mode=1,save=0):
    """

    :param dataset: training dataset
    :param epochs: number of training epochs
    :param mode: training mode 1 - full mode 2 - both networks trained separate 3 - only "small" network 4 - only "big" network
    :param save: control over saving checkpoints
    :return:
    """
    if mode ==1:
      for epoch in range(epochs):
        start = time.time()
        batch = 0
        images = []
        for image_batch in dataset:
          print(f"batch {batch}")
          train_step_both(image_batch)
          batch = batch + 1
          images = image_batch[:4]
        images = tf.image.resize(images,[32,32])
        tmp = generate_and_save_images(generator1,
                                 epoch + 1,
                                 seed)
        generate_and_save_images(generator2,
                                 epoch,
                                 tmp,"big_from_small")
        if save:
         if (epoch + 1) % 15 == 0:
           checkpoint1.save(file_prefix=checkpoint_prefix_1)
           checkpoint2.save(file_prefix = checkpoint_prefix_2)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      display.clear_output(wait=True)
      tmp = generate_and_save_images(generator1,
                               epochs,
                               seed)
      generate_and_save_images(generator2,
                               epochs,
                               tmp, "big_frolm_small")
    if mode ==2:
        for epoch in range(epochs):
            start = time.time()
            batch = 0
            images = []
            for image_batch in dataset:
                print(f"batch {batch}")
                train_step_both_separate(image_batch)
                batch = batch + 1
                images = image_batch[:4]
            images = tf.image.resize(images, [32, 32])
            tmp = generate_and_save_images(generator1,
                                           epoch + 1,
                                           seed)
            generate_and_save_images(generator2,
                                     epoch,
                                     images, "big_from_clear")
            if save:
                if (epoch + 1) % 15 == 0:
                    checkpoint1.save(file_prefix=checkpoint_prefix_1)
                    checkpoint2.save(file_prefix=checkpoint_prefix_2)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        display.clear_output(wait=True)
        tmp = generate_and_save_images(generator1,
                                       epochs,
                                       seed)
        generate_and_save_images(generator2,
                                 epochs,
                                 images, "big_from_clear")
    if mode == 3:
        for epoch in range(epochs):
            start = time.time()
            batch = 0
            images = []
            for image_batch in dataset:
                print(f"batch {batch}")
                train_step_only1(image_batch)
                batch = batch + 1
                images = image_batch[:4]
            images = tf.image.resize(images, [32, 32])
            tmp = generate_and_save_images(generator1,
                                           epoch + 1,
                                           seed,"only1")
            if save:
                if (epoch + 1) % 15 == 0:
                    checkpoint1.save(file_prefix=checkpoint_prefix_1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        display.clear_output(wait=True)
        tmp = generate_and_save_images(generator1,
                                       epochs,
                                       seed,"only1")
    if mode == 4:
        for epoch in range(epochs):
            start = time.time()
            batch = 0
            images = []
            for image_batch in dataset:
                print(f"batch {batch}")
                train_step_only2(image_batch)
                batch = batch + 1
                images = image_batch[:4]
            images = tf.image.resize(images, [32, 32])
            generate_and_save_images(generator2,
                                     epoch,
                                     images, "only2")
            if save:
                if (epoch + 1) % 15 == 0:
                    checkpoint2.save(file_prefix=checkpoint_prefix_2)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        display.clear_output(wait=True)
        generate_and_save_images(generator2,
                                 epochs,
                                 images, "only2")

def generate_and_save_images(model, epoch, test_input,text=""):
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
      plt.subplot(2, 2, i+1)
      plt.imshow(predictions[i, :, :, 0])
      plt.axis('off')
  plt.savefig(f'image_at_epoch_{epoch}_{text}.png')
  return predictions


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
    num_generated = 1024
    num_elements = 1024
    EPOCHS = 150
    noise_dim = 25
    num_examples_to_generate = 4
    save = True
    mode = 3
    load = True
    train_images2 = random_data(num_generated,x)
    test = list(np.random.choice(num_generated,num_elements))
    train_images = []
    for i in test:
        train_images.append(train_images2[i])
    BUFFER_SIZE = num_elements
    BATCH_SIZE = 32
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                  discriminator_optimizer=discriminator_optimizer,
    #                                  generator1=generator1,
    #                                  generator2=generator2,
    #                                  discriminator1=discriminator1,
    #                                  discriminator2=discriminator2)

    if mode != 4:
        generator1 = generator_1(int(x/4),int(y/4),z)
        discriminator1 = discriminator_1(32,32,3)
        checkpoint_dir = './training_checkpoints_1'
        checkpoint_prefix_1 = os.path.join(checkpoint_dir, "ckpt")
        checkpoint1 = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator1=generator1,
                                         discriminator1=discriminator1)
        if load:
            checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir))

    if mode != 3:
        generator2 = generator_2(x, y, z)
        discriminator2 = discriminator_2(128, 128, 3)
        checkpoint_dir = './training_checkpoints_2'
        checkpoint_prefix_2 = os.path.join(checkpoint_dir, "ckpt")
        checkpoint2 = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator2=generator2,
                                         discriminator2=discriminator2)
        if load:
            checkpoint2.restore(tf.train.latest_checkpoint(checkpoint_dir))


    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    train(train_dataset, EPOCHS,mode,save)
