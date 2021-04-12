import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError
import random
import time
from datetime import datetime
import argparse
import os

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
from pixel_shuffler import PixelShuffler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# from tensorflow.python.framework.ops import disable_eager_execution
#
# disable_eager_execution()


# ********************************************************************

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024

latent_dim = 128

# optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

# ********************************************************************

LEARNING_RATE = 0.0002
BETA_1 = 0.5
LAMBDA_CYCLE = 10.0
LAMBDA_ID = 5
POOL_SIZE = 50
EPOCHS = 200
DECAY_EPOCHS = 100
SHUFFLE_SIZE = 10000


# ********************************************************************

class LinearDecay(tf.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero
    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                    1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self._initial_learning_rate,
            'total_steps': self._steps,
            'step_decay': self._step_decay,
        }


# This image pool only works in TF eager mode. Not graph mode (tf.function)
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.count = 0
        self.pool = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            # if the buffer is not full; keep inserting current images to the buffer
            if self.count < self.pool_size:
                self.count = self.count + 1
                self.pool.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # by 50% chance, the buffer will return a previously stored image
                    # and insert the current image into the buffer
                    # randint is inclusive
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.pool[random_id]
                    self.pool[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return tf.stack(return_images, axis=0)


# ********************************************************************

class ReflectionPad2d(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        super(ReflectionPad2d, self).__init__(**kwargs)
        self.padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.padding, 'REFLECT')


# ********************************************************************

class ResNetBlock(tf.keras.Model):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.padding1 = ReflectionPad2d(1)
        self.conv1 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.padding2 = ReflectionPad2d(1)
        self.conv2 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.padding1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.padding2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        outputs = inputs + x
        return outputs


# ********************************************************************

def make_generator_model(n_blocks):
    # 6 residual blocks
    # c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
    # 9 residual blocks
    # c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
    model = tf.keras.Sequential()

    # Encoding
    model.add(ReflectionPad2d(3, input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(1, 1), padding='valid', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Transformation
    for i in range(n_blocks):
        model.add(ResNetBlock(256))

    # Decoding
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(ReflectionPad2d(3))
    model.add(tf.keras.layers.Conv2D(3, (7, 7), strides=(1, 1), padding='valid', activation='tanh'))

    return model


# ********************************************************************

def make_discriminator_model():
    # C64-C128-C256-C512
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # This last conv net is the PatchGAN
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39#issuecomment-305575964
    # https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    model.add(tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same'))

    return model


# ********************************************************************

def main():
    parser = argparse.ArgumentParser(
        description='Convert TFRecords for CycleGAN dataset.')
    parser.add_argument(
        '--dataset', help='The name of the dataset', required=False)
    parser.add_argument(
        '--batch_size', help='The batch size of input data', default='4')
    args = parser.parse_args()

    loss_gen_total_metrics = tf.keras.metrics.Mean(
        'loss_gen_total_metrics', dtype=tf.float32)
    loss_dis_total_metrics = tf.keras.metrics.Mean(
        'loss_dis_total_metrics', dtype=tf.float32)
    loss_cycle_a2b2a_metrics = tf.keras.metrics.Mean(
        'loss_cycle_a2b2a_metrics', dtype=tf.float32)
    loss_cycle_b2a2b_metrics = tf.keras.metrics.Mean(
        'loss_cycle_b2a2b_metrics', dtype=tf.float32)
    loss_gen_a2b_metrics = tf.keras.metrics.Mean(
        'loss_gen_a2b_metrics', dtype=tf.float32)
    loss_gen_b2a_metrics = tf.keras.metrics.Mean(
        'loss_gen_b2a_metrics', dtype=tf.float32)
    loss_dis_b_metrics = tf.keras.metrics.Mean(
        'loss_dis_b_metrics', dtype=tf.float32)
    loss_dis_a_metrics = tf.keras.metrics.Mean(
        'loss_dis_a_metrics', dtype=tf.float32)
    loss_id_b2a_metrics = tf.keras.metrics.Mean(
        'loss_id_b2a_metrics', dtype=tf.float32)
    loss_id_a2b_metrics = tf.keras.metrics.Mean(
        'loss_id_a2b_metrics', dtype=tf.float32)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    fake_pool_b2a = ImagePool(POOL_SIZE)
    fake_pool_a2b = ImagePool(POOL_SIZE)

    # ********************************************************************

    def calc_gan_loss(prediction, is_real):
        # Typical GAN loss to set objectives for generator and discriminator
        if is_real:
            return mse_loss(prediction, tf.ones_like(prediction))
        else:
            return mse_loss(prediction, tf.zeros_like(prediction))

    def calc_cycle_loss(reconstructed_images, real_images):
        # Cycle loss to make sure reconstructed image looks real
        return mae_loss(reconstructed_images, real_images)

    def calc_identity_loss(identity_images, real_images):
        # Identity loss to make sure generator won't do unnecessary change
        # Ideally, feeding a real image to generator should generate itself
        return mae_loss(identity_images, real_images)

    # ********************************************************************

    def make_dataset(filepath):
        raw_dataset = tf.data.TFRecordDataset(filepath)

        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }

        def preprocess_image(encoded_image):
            image = tf.image.decode_jpeg(encoded_image, 3)
            # random flip left or right
            image = tf.image.random_flip_left_right(image)
            # resize to 286x286
            image = tf.image.resize(image, [286, 286])
            # random crop a 256x256 area
            image = tf.image.random_crop(
                image, [256, 256, tf.shape(image)[-1]])
            # normalize from 0-255 to -1 ~ +1
            image = image / 127.5 - 1
            return image

        def parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            features = tf.io.parse_single_example(example_proto,
                                                  image_feature_description)
            encoded_image = features['image/encoded']
            image = preprocess_image(encoded_image)
            return image

        parsed_image_dataset = raw_dataset.map(parse_image_function)
        return parsed_image_dataset

    def count_dataset_batches(dataset):
        size = 0
        for _ in dataset:
            size += 1
        return size

    # ************************************************************************

    batch_size = 2

    images_A = get_image_paths("data/oliwka_256")
    images_B = get_image_paths("data/laura_256")

    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0

    warped_A, target_A = get_training_data(images_A, batch_size, 256, 4)
    warped_B, target_B = get_training_data(images_B, batch_size, 256, 4)

    # for local testing
    seed1 = tf.random.normal([2, 256, 256, 3])
    seed2 = tf.random.normal([2, 256, 256, 3])

    seed1 = tf.convert_to_tensor(target_A)
    seed2 = tf.convert_to_tensor(target_B)

    combined_dataset = [(seed1, seed2)]
    EPOCHS = 2

    # ************************************************************************

    # train_a = make_dataset('tfrecords/{}/trainA.tfrecord'.format(args.dataset))
    # train_b = make_dataset('tfrecords/{}/trainB.tfrecord'.format(args.dataset))
    # combined_dataset = tf.data.Dataset.zip(
    #     (train_a, train_b)).shuffle(SHUFFLE_SIZE).batch(int(args.batch_size))
    total_batches = count_dataset_batches(combined_dataset)
    print('Batch size: {}, Total batches per epoch: {}'.format(
        args.batch_size, total_batches))

    # ************************************************************************

    generator_a2b = make_generator_model(n_blocks=9)
    generator_b2a = make_generator_model(n_blocks=9)
    discriminator_b = make_discriminator_model()
    discriminator_a = make_discriminator_model()

    gen_lr_scheduler = LinearDecay(LEARNING_RATE, EPOCHS * total_batches,
                                   DECAY_EPOCHS * total_batches)
    dis_lr_scheduler = LinearDecay(LEARNING_RATE, EPOCHS * total_batches,
                                   DECAY_EPOCHS * total_batches)
    optimizer_gen = tf.keras.optimizers.Adam(gen_lr_scheduler, BETA_1)
    optimizer_dis = tf.keras.optimizers.Adam(dis_lr_scheduler, BETA_1)

    checkpoint_dir = './checkpoints-{}'.format('oliwkaTolaura')
    checkpoint = tf.train.Checkpoint(
        generator_a2b=generator_a2b,
        generator_b2a=generator_b2a,
        discriminator_b=discriminator_b,
        discriminator_a=discriminator_a,
        optimizer_gen=optimizer_gen,
        optimizer_dis=optimizer_dis,
        epoch=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=None)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # ************************************************************************

    @tf.function
    def train_generator(images_a, images_b):
        real_a = images_a
        real_b = images_b

        with tf.GradientTape() as tape:
            # Cycle A -> B -> A
            fake_a2b = generator_a2b(real_a, training=True)
            recon_b2a = generator_b2a(fake_a2b, training=True)
            # Cycle B -> A -> B
            fake_b2a = generator_b2a(real_b, training=True)
            recon_a2b = generator_a2b(fake_b2a, training=True)

            # Use real B to generate B should be identical
            identity_a2b = generator_a2b(real_b, training=True)
            identity_b2a = generator_b2a(real_a, training=True)
            loss_identity_a2b = calc_identity_loss(identity_a2b, real_b)
            loss_identity_b2a = calc_identity_loss(identity_b2a, real_a)

            # Generator A2B tries to trick Discriminator B that the generated image is B
            loss_gan_gen_a2b = calc_gan_loss(
                discriminator_b(fake_a2b, training=True), True)
            # Generator B2A tries to trick Discriminator A that the generated image is A
            loss_gan_gen_b2a = calc_gan_loss(
                discriminator_a(fake_b2a, training=True), True)
            loss_cycle_a2b2a = calc_cycle_loss(recon_b2a, real_a)
            loss_cycle_b2a2b = calc_cycle_loss(recon_a2b, real_b)

            # Total generator loss
            loss_gen_total = loss_gan_gen_a2b + loss_gan_gen_b2a \
                             + (loss_cycle_a2b2a + loss_cycle_b2a2b) * LAMBDA_CYCLE \
                             + (loss_identity_a2b + loss_identity_b2a) * LAMBDA_ID

        trainable_variables = generator_a2b.trainable_variables + generator_b2a.trainable_variables
        gradient_gen = tape.gradient(loss_gen_total, trainable_variables)
        optimizer_gen.apply_gradients(zip(gradient_gen, trainable_variables))

        # Metrics
        loss_gen_a2b_metrics(loss_gan_gen_a2b)
        loss_gen_b2a_metrics(loss_gan_gen_b2a)
        loss_id_b2a_metrics(loss_identity_b2a)
        loss_id_a2b_metrics(loss_identity_a2b)
        loss_cycle_a2b2a_metrics(loss_cycle_a2b2a)
        loss_cycle_b2a2b_metrics(loss_cycle_b2a2b)
        loss_gen_total_metrics(loss_gen_total)

        loss_dict = {
            'loss_gen_a2b': loss_gan_gen_a2b,
            'loss_gen_b2a': loss_gan_gen_b2a,
            'loss_id_a2b': loss_identity_a2b,
            'loss_id_b2a': loss_identity_b2a,
            'loss_cycle_a2b2a': loss_cycle_a2b2a,
            'loss_cycle_b2a2b': loss_cycle_b2a2b,
            'loss_gen_total': loss_gen_total,
        }
        return fake_a2b, fake_b2a, loss_dict

    # ********************************************************************

    @tf.function
    def train_discriminator(images_a, images_b, fake_a2b, fake_b2a):
        real_a = images_a
        real_b = images_b

        with tf.GradientTape() as tape:
            # Discriminator A should classify real_a as A
            loss_gan_dis_a_real = calc_gan_loss(
                discriminator_a(real_a, training=True), True)
            # Discriminator A should classify generated fake_b2a as not A
            loss_gan_dis_a_fake = calc_gan_loss(
                discriminator_a(fake_b2a, training=True), False)

            # Discriminator B should classify real_b as B
            loss_gan_dis_b_real = calc_gan_loss(
                discriminator_b(real_b, training=True), True)
            # Discriminator B should classify generated fake_a2b as not B
            loss_gan_dis_b_fake = calc_gan_loss(
                discriminator_b(fake_a2b, training=True), False)

            # Total discriminator loss
            loss_dis_a = (loss_gan_dis_a_real + loss_gan_dis_a_fake) * 0.5
            loss_dis_b = (loss_gan_dis_b_real + loss_gan_dis_b_fake) * 0.5
            loss_dis_total = loss_dis_a + loss_dis_b

        trainable_variables = discriminator_a.trainable_variables + discriminator_b.trainable_variables
        gradient_dis = tape.gradient(loss_dis_total, trainable_variables)
        optimizer_dis.apply_gradients(zip(gradient_dis, trainable_variables))

        # Metrics
        loss_dis_a_metrics(loss_dis_a)
        loss_dis_b_metrics(loss_dis_b)
        loss_dis_total_metrics(loss_dis_total)

        return {
            'loss_dis_b': loss_dis_b,
            'loss_dis_a': loss_dis_a,
            'loss_dis_total': loss_dis_total
        }

    # ********************************************************************

    def train_step(images_a, images_b, epoch, step):
        fake_a2b, fake_b2a, gen_loss_dict = train_generator(images_a, images_b)

        fake_b2a_from_pool = fake_pool_b2a.query(fake_b2a)
        fake_a2b_from_pool = fake_pool_a2b.query(fake_a2b)

        dis_loss_dict = train_discriminator(
            images_a, images_b, fake_a2b_from_pool, fake_b2a_from_pool)

        gen_loss_list = [
            '{}:{} '.format(k, v) for k, v in gen_loss_dict.items()
        ]
        dis_loss_list = [
            '{}:{} '.format(k, v) for k, v in dis_loss_dict.items()
        ]

        tf.print('Epoch {} Step {} '.format(epoch, step),
                 ' '.join(gen_loss_list + dis_loss_list))

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/{}/{}/train'.format(args.dataset, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # ********************************************************************

    def write_metrics(epoch):
        with train_summary_writer.as_default():
            tf.summary.scalar(
                'loss_gen_a2b', loss_gen_a2b_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_gen_b2a', loss_gen_b2a_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_dis_b', loss_dis_b_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_dis_a', loss_dis_a_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_id_a2b', loss_id_a2b_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_id_b2a', loss_id_b2a_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_gen_total', loss_gen_total_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_dis_total', loss_dis_total_metrics.result(), step=epoch)
            tf.summary.scalar(
                'loss_cycle_a2b2a',
                loss_cycle_a2b2a_metrics.result(),
                step=epoch)
            tf.summary.scalar(
                'loss_cycle_b2a2b',
                loss_cycle_b2a2b_metrics.result(),
                step=epoch)
            tf.summary.scalar(
                'gen_learning_rate',
                gen_lr_scheduler.current_learning_rate,
                step=epoch)
            tf.summary.scalar(
                'dis_learning_rate',
                dis_lr_scheduler.current_learning_rate,
                step=epoch)

        loss_gen_a2b_metrics.reset_states()
        loss_gen_b2a_metrics.reset_states()
        loss_dis_b_metrics.reset_states()
        loss_dis_a_metrics.reset_states()
        loss_id_a2b_metrics.reset_states()
        loss_id_b2a_metrics.reset_states()
        return

    # ********************************************************************

    def train(dataset, epochs):

        for epoch in range(checkpoint.epoch + 1, epochs + 1):
            start = time.time()
            print('Epoch {} starts. Learning rate: {}, {}'.format(
                epoch, gen_lr_scheduler.current_learning_rate,
                dis_lr_scheduler.current_learning_rate))

            # Training
            for (step, batch) in enumerate(dataset):
                train_step(batch[0], batch[1], epoch, step)

            # Update TensorBoard metrics
            write_metrics(epoch)

            # Save checkpoint
            checkpoint.epoch.assign_add(1)
            if epoch % 2 == 0:
                save_path = checkpoint_manager.save()
                print("Saved checkpoint for epoch {}: {}".format(
                    int(checkpoint.epoch), save_path))

            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

    train(combined_dataset, EPOCHS)
    print('Finished training.')


# ********************************************************************

if __name__ == '__main__':
    main()

# ********************************************************************

# images_A = get_image_paths("data/oliwka_256")
# images_B = get_image_paths("data/laura_256")
# images_A = load_images(images_A) / 255.0
# images_B = load_images(images_B) / 255.0
#
# images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
#
# for epoch in range(100000):
#     batch_size = 8
#
#     warped_A, target_A = get_training_data(images_A, batch_size, 256, 4)
#     warped_B, target_B = get_training_data(images_B, batch_size, 256, 4)
#
#     loss_A = cycle_gan_model.train_on_batch(target_A, target_B)
#     print(epoch, loss_A)
