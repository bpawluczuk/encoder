import datetime
import cv2
import numpy
import seaborn as sns

sns.set()

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow_addons as tfa
from tensorflow.keras import layers

from lib.utils import get_image_paths, load_images, stack_images
from lib.training_data import get_training_data
from lib.pixel_shuffler import PixelShuffler

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

devices = session.list_devices()
for d in devices:
    print(d.name)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# ********************************************************************

size = 256
width = 256
height = 256
chanels = 3

IMAGE_SHAPE = (size, size, chanels)
ENCODER_DIM = 1024

_latent_dim = 512  # 128
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=2e-5, beta_1=0.5)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
):
    def block(x):
        filters = x.shape[-1]
        input_tensor = x

        # x = ReflectionPadding2D()(input_tensor)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)

        # x = ReflectionPadding2D()(x)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = layers.add([input_tensor, x])

        return x

    return block


# ********************************************************************************

def downsample(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=True,
        dropout_rate=0,
        use_bias=False,
):
    def block(x):
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x

    return block


# ********************************************************************************

def upsampleTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=True,
        use_bias=False,
):
    def block(x):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_init,
            use_bias=use_bias,
        )(x)

        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        return x

    return block


def upsampleShuffler(
        filters,
        kernel_size=3,
        padding='same',
        activation=True,
        use_bias=False
):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init,
            use_bias=use_bias,
        )(x)

        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        x = PixelShuffler()(x)

        return x

    return block


# ********************************************************************************

def get_resnet_generator(
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=12,
        name='resnet_generator',
):
    input_ = layers.Input(shape=IMAGE_SHAPE)

    x = downsample(64, kernel_size=5, strides=1)(input_)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(filters=filters)(x)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block()(x)

    filters *= 2
    x = upsampleShuffler(filters)(x)
    filters *= 2
    x = upsampleShuffler(filters)(x)

    # Final block
    x = layers.Conv2D(3, (5, 5), padding="same")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(input_, x, name=name)
    model.summary()
    return model


# ********************************************************************************

def get_discriminator(filters=64, name='discriminator'):
    input_ = layers.Input(shape=IMAGE_SHAPE)

    x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(input_)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters

    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(2, 2))(x)
    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(2, 2))(x)
    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(1, 1))(x)

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_init)(x)

    model = keras.models.Model(inputs=input_, outputs=x, name=name)
    return model


# ********************************************************************************

lambda_cycle = 10.0
lambda_id = 0.5

disc_A = get_discriminator(name="disc_A")
disc_B = get_discriminator(name="disc_B")

disc_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
disc_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

gen_AB = get_resnet_generator(name="gen_AB")
gen_BA = get_resnet_generator(name="gen_BA")

img_A = Input(shape=IMAGE_SHAPE)
img_B = Input(shape=IMAGE_SHAPE)

fake_B = gen_AB(img_A)
fake_A = gen_BA(img_B)

reconstr_A = gen_BA(fake_B)
reconstr_B = gen_AB(fake_A)

same_A = gen_BA(img_A)
same_B = gen_AB(img_B)

disc_A.trainable = False
disc_B.trainable = False

valid_A = disc_A(fake_A)
valid_B = disc_B(fake_B)

# ********************************************************************************

cyclegan = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, same_A, same_B])

cyclegan.compile(
    loss=['mae', 'mae', 'mse', 'mse', 'mse', 'mse'],
    loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
    optimizer=optimizer
)

disc_A.summary()
gen_AB.summary()
cyclegan.summary()

# ********************************************************************

try:
    cyclegan.load_weights("models/CycleGan/cyclegan_batch.h5")
    print("... load model")
except:
    print("model does not exist")


def save_model_weights():
    cyclegan.save_weights("models/CycleGan/cyclegan_batch.h5")
    print("save model weights")


# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW/trainOL")
images_B = get_image_paths("data_train/LU_NEW/trainLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

batch_size = 1
epochs = 100
dataset_size = len(images_A)
batches = round(dataset_size / batch_size)
plot_result_test = 1000
save_interval = 1000
sample_interval = 10

# ********************************************************************************

test_images_A = get_image_paths("data_train/OL_NEW/testOL")
test_images_B = get_image_paths("data_train/LU_NEW/testLU")

valid_images_A = get_image_paths("data_train/OL_NEW/validOL")
valid_images_B = get_image_paths("data_train/LU_NEW/validLU")

epoch_loss_history_disc = []
epoch_loss_history_gen = []
epoch_acc_history_disc = []
epoch_acc_history_gen = []

avg_index = []
avg_history_loss_disc = []
avg_history_loss_gen = []
avg_history_acc_disc = []
avg_history_acc_gen = []

history_dir = 'history/GAN/'
stats_loss_disc = history_dir + 'stats_loss_disc.txt'
stats_acc_disc = history_dir + 'stats_acc_disc.txt'
stats_loss_gen = history_dir + 'stats_loss_gen.txt'
stats_acc_gen = history_dir + 'stats_acc_gen.txt'
stats_time = history_dir + 'stats_time.txt'

# ********************************************************************************

start_time = datetime.datetime.now()

valid = numpy.ones((batch_size,) + (32, 32, 1))
fake = numpy.zeros((batch_size,) + (32, 32, 1))

for epoch in range(epochs):
    epoch += 1

    start_epoch_time = datetime.datetime.now()

    for batch in range(batches):
        batch += 1

        warped_A, target_A = get_training_data(images_A, batch_size, size, zoom)
        warped_B, target_B = get_training_data(images_B, batch_size, size, zoom)

        fake_B = gen_AB.predict(target_A)
        fake_A = gen_BA.predict(target_B)

        dA_loss_real = disc_A.train_on_batch(target_A, valid)
        dA_loss_fake = disc_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * numpy.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = disc_B.train_on_batch(target_B, valid)
        dB_loss_fake = disc_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * numpy.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * numpy.add(dA_loss, dB_loss)

        # Train the generators
        g_loss = cyclegan.train_on_batch([target_A, target_B], [valid, valid, target_A, target_B, target_A, target_B])

        elapsed_time = datetime.datetime.now() - start_time

        if 1:
            epoch_loss_history_disc.append(d_loss[0])
            epoch_acc_history_disc.append(100 * d_loss[1])
            epoch_loss_history_gen.append(g_loss[0])
            epoch_acc_history_gen.append(numpy.mean(g_loss[1:3]))

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, same: %05f] time: %s " \
            % (epoch, epochs,
               batch, batches,
               d_loss[0], 100 * d_loss[1],
               g_loss[0],
               numpy.mean(g_loss[1:3]),
               numpy.mean(g_loss[3:5]),
               numpy.mean(g_loss[5:6]),
               elapsed_time))

        if batch % save_interval == 0:
            save_model_weights()

        if batch % sample_interval == 0:
            test_A = target_A[0:3]
            test_B = target_B[0:3]

            fake_B = gen_AB.predict(test_A)
            fake_A = gen_BA.predict(test_B)

            reconstr_A = gen_BA.predict(fake_B)
            reconstr_B = gen_AB.predict(fake_A)

            figure = numpy.stack([
                test_A,
                fake_B,
                reconstr_A,
                test_B,
                fake_A,
                reconstr_B,
            ], axis=1)

            figure = numpy.concatenate([figure], axis=0)
            figure = stack_images(figure)

            figure = numpy.clip(figure * 255, 0, 255).astype(numpy.uint8)

            cv2.imshow("Results", figure)
            key = cv2.waitKey(1)

        if batch % batches == 0:

            _, ax = plt.subplots(4, 4, figsize=(16, 16))

            for i, fn in enumerate(valid_images_A):
                test_image = cv2.imread(fn)
                test_image_tensor = numpy.expand_dims(test_image, 0)
                predict_image = gen_AB.predict(test_image_tensor)

                ax[i, 0].imshow(cv2.cvtColor(test_image_tensor[0], cv2.COLOR_BGR2RGB))
                ax[i, 1].imshow(cv2.cvtColor(predict_image[0], cv2.COLOR_BGR2RGB))
                ax[i, 0].set_title("Osoba A")
                ax[i, 1].set_title("Osoba B")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")

            for i, fn in enumerate(valid_images_B):
                test_image = cv2.imread(fn)
                test_image_tensor = numpy.expand_dims(test_image, 0)
                predict_image = gen_BA.predict(test_image_tensor)

                ax[i, 2].imshow(cv2.cvtColor(test_image_tensor[0], cv2.COLOR_BGR2RGB))
                ax[i, 3].imshow(cv2.cvtColor(predict_image[0], cv2.COLOR_BGR2RGB))
                ax[i, 2].set_title("Osoba B")
                ax[i, 3].set_title("Osoba A")
                ax[i, 2].axis("off")
                ax[i, 3].axis("off")

            plt.show()
            plt.savefig(history_dir + "predict_" + str(epoch).zfill(3) + ".jpg")
            plt.clf()
            plt.close()

        if batch % 2 == 0:

            # ------- Epoch time ------------------

            end_epoch_time = datetime.datetime.now() - start_epoch_time
            with open(stats_time, "a+") as f:
                f.write(str(end_epoch_time) + "\n")
                f.close()

            # ------- Epoch index -----------------

            avg_index.append(len(avg_index) + 1)

            # ------- Loss discriminator ----------------

            loss_sum = 0
            for loss in epoch_loss_history_disc:
                loss_sum += loss

            avg_loss = loss_sum / len(epoch_loss_history_disc)
            avg_history_loss_disc.append(avg_loss)

            with open(stats_loss_disc, "a+") as f:
                f.write(str(avg_loss) + "\n")
                f.close()

            epoch_loss_history_disc = []

            # plt.scatter(avg_index, avg_history_loss_disc, s=20, label="Discriminator loss")
            # plt.legend()
            # plt.show()
            # plt.savefig(history_dir + "loss_disc_" + str(epoch).zfill(3) + "_plot.jpg")

            # ------- Accuracy discriminator -------

            acc_sum = 0
            for acc in epoch_acc_history_disc:
                acc_sum += acc

            avg_acc = acc_sum / len(epoch_acc_history_disc)
            avg_history_acc_disc.append(avg_acc)

            with open(stats_acc_disc, "a+") as f:
                f.write(str(avg_acc) + "\n")
                f.close()

            epoch_acc_history_disc = []

            # plt.scatter(avg_index, avg_history_acc_disc, s=20, label="Discriminator accuracy")
            # plt.legend()
            # plt.show()
            # plt.savefig(history_dir + "acc_disc_" + str(epoch).zfill(3) + "_plot.jpg")

            # ------- Loss generator ----------------

            loss_sum = 0
            for loss in epoch_loss_history_gen:
                loss_sum += loss

            avg_loss = loss_sum / len(epoch_loss_history_gen)
            avg_history_loss_gen.append(avg_loss)

            with open(stats_loss_gen, "a+") as f:
                f.write(str(avg_loss) + "\n")
                f.close()

            epoch_loss_history_gen = []

            # plt.scatter(avg_index, avg_history_loss_gen, s=20, label="Generator loss")
            # plt.legend()
            # plt.show()
            # plt.savefig(history_dir + "loss_gen_" + str(epoch).zfill(3) + "_plot.jpg")

            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(avg_index, avg_history_loss_disc, label="Discriminator loss")
            axs[0, 0].set_title('Axis [0, 0]')
            axs[0, 1].plot(avg_index, avg_history_acc_disc, label="Discriminator accuracy")
            axs[0, 1].set_title('Axis [0, 1]')
            axs[1, 0].plot(avg_index, avg_history_loss_gen, label="Generator loss")
            axs[1, 0].set_title('Axis [1, 0]')
            plt.show()