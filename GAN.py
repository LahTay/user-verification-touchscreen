
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

# mnist for testing
# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = (train_images[0:7500].astype('float16') - 127.5) / 127.5
# train_images = np.expand_dims(train_images, axis=-1)
# train_images = tf.image.resize(train_images, [140,140])
@timebudget
def run_generations(operation, inputs, pool):
    return pool.map(operation, inputs)

def random_data(x):
    processes_pool = Pool(6)
    inputs = np.ones(x).astype(int)*100
    out = run_generations(TS_option_for_preprocessing.generate, inputs, processes_pool)
    return out

def make_generator_model(x=100,y=100,z=3):
    #magic
    # TODO: make better model simpler to understand
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(x/4)*int(y/4)*64, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(x/4), int(y/4), 64)))
    assert model.output_shape == (None, int(x/4), int(y/4), 64)

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

def alt_make_generator_model():
    #magic
    # TODO: make better model simpler to understand
    model = tf.keras.Sequential()
    model.add(layers.Dense(35*35*64, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((35, 35, 64)))
    assert model.output_shape == (None, 35, 35, 64)

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 35, 35, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 70, 70, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 140, 140, 1)
    model.summary()
    return model

def make_discriminator_model():
    # magic 2
    # TODO: make better model simpler to understand
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[100, 100, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
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
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    batch = 0
    for image_batch in dataset:
      print(f"batch {batch}")
      train_step(image_batch)
      batch = batch + 1

    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
      plt.subplot(2, 2, i+1)
      plt.imshow(predictions[i, :, :, 0])
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


if __name__ == '__main__':
    train_images = random_data(2000)
    BUFFER_SIZE = 2000
    BATCH_SIZE = 128
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    generator = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print(decision)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 4

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    train(train_dataset, EPOCHS)
