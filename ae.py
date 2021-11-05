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

_latent_dim = 256  # 128
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=2e-5, beta_1=0.5)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

def downsample(
        filters,
        kernel_size=4,
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


def upsampleShuffler(filters, kernel_size=4, filter_times=2, padding='same', activation=True):
    def block(x):
        x = Conv2D(
            filters * filter_times,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init
        )(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        x = PixelShuffler()(x)

        return x

    return block


def upsampleTranspose(
        filters,
        kernel_size=4,
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


def Encoder(input_, name="Encoder"):
    x = downsample(64, kernel_size=5)(input_)
    x = downsample(128)(x)
    x = downsample(256)(x)
    x = downsample(512)(x)
    x = Flatten()(x)

    latent_space = Dense(_latent_dim)(x)

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)

    filters = 512
    x = upsampleShuffler(filters)(x)

    return Model(input_, x, name=name)


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 256))

    x = upsampleShuffler(256)(input_)
    x = upsampleShuffler(128)(x)
    x = upsampleShuffler(64, filter_times=4)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

    return Model(input_, x, name=name)


# ********************************************************************

x = Input(shape=IMAGE_SHAPE)

encoder = Encoder(x)
decoder_A = Decoder()
decoder_B = Decoder()

# ********************************************************************

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

encoder.summary()
# autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

try:
    encoder.load_weights("models/AE/encoder.h5")
    decoder_A.load_weights("models/AE/decoder_a.h5")
    decoder_B.load_weights("models/AE/decoder_b.h5")
    print("... load model")
except:
    print("model does not exist")


def save_model_weights():
    encoder.save_weights("models/AE/encoder.h5")
    decoder_A.save_weights("models/AE/decoder_a.h5")
    decoder_B.save_weights("models/AE/decoder_b.h5")
    print("save model weights")


# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW/trainOL")
images_B = get_image_paths("data_train/LU_NEW/trainLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

# ********************************************************************************

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

epoch_loss_history_encoder = []
epoch_acc_history_encoder = []

test_epoch_loss_history_encoder = []
test_epoch_acc_history_encoder = []

avg_index = []
avg_history_loss = []
avg_history_acc = []

test_avg_index = []
test_avg_history_loss = []
test_avg_history_acc = []

history_dir = 'history/AE/'
stats_loss = history_dir + 'stats_loss.txt'
stats_acc = history_dir + 'stats_acc.txt'

# ********************************************************************************

start_time = datetime.datetime.now()

inc = 1
for epoch in range(epochs):
    epoch += 1
    for batch in range(batches):
        batch += 1

        warped_A, target_A = get_training_data(images_A, batch_size, size, zoom)
        warped_B, target_B = get_training_data(images_B, batch_size, size, zoom)

        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)

        if epoch != 1:
            # Epoch encoder loss
            epoch_loss_history_encoder.append(0.5 * (loss_A[0] + loss_B[0]))
            # Epoch encoder acc
            epoch_acc_history_encoder.append(0.5 * (loss_A[1] + loss_B[1]))

        elapsed_time = datetime.datetime.now() - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [A loss: %f, acc: %3d%%] [B loss: %f, acc: %3d%%] time: %s " \
            % (epoch, epochs,
               batch, batches,
               loss_A[0], 100 * loss_A[1],
               loss_B[0], 100 * loss_B[1],
               elapsed_time))

        if batch % save_interval == 0:
            save_model_weights()

        if batch % sample_interval == 0:
            test_A = target_A[0:3]
            test_B = target_B[0:3]

            figure = numpy.stack([
                test_A,
                autoencoder_A.predict(test_A),
                autoencoder_B.predict(test_A),
                test_B,
                autoencoder_B.predict(test_B),
                autoencoder_A.predict(test_B),
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
                predict_image = autoencoder_B.predict(test_image_tensor)

                ax[i, 0].imshow(cv2.cvtColor(test_image_tensor[0], cv2.COLOR_BGR2RGB))
                ax[i, 1].imshow(cv2.cvtColor(predict_image[0], cv2.COLOR_BGR2RGB))
                ax[i, 0].set_title("Osoba A")
                ax[i, 1].set_title("Osoba B")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")

            for i, fn in enumerate(valid_images_B):
                test_image = cv2.imread(fn)
                test_image_tensor = numpy.expand_dims(test_image, 0)
                predict_image = autoencoder_A.predict(test_image_tensor)

                ax[i, 2].imshow(cv2.cvtColor(test_image_tensor[0], cv2.COLOR_BGR2RGB))
                ax[i, 3].imshow(cv2.cvtColor(predict_image[0], cv2.COLOR_BGR2RGB))
                ax[i, 2].set_title("Osoba B")
                ax[i, 3].set_title("Osoba A")
                ax[i, 2].axis("off")
                ax[i, 3].axis("off")

            plt.clf()
            plt.savefig(history_dir + "predict_" + str(epoch).zfill(3) + ".jpg")
            plt.show()
            plt.close()

        if batch % batches == 0 and epoch != 1:

            avg_index.append(len(avg_index) + 1)

            # -------

            loss_sum = 0
            for loss in epoch_loss_history_encoder:
                loss_sum += loss

            avg_loss = loss_sum / len(epoch_loss_history_encoder)
            avg_history_loss.append(avg_loss)

            with open(stats_loss, "a+") as f:
                f.write(str(avg_loss) + "\n")
                f.close()

            epoch_loss_history_encoder = []

            # -------

            plt.clf()
            plt.scatter(avg_index, avg_history_loss, s=20, label="Encoder loss")
            plt.legend()
            plt.show()
            plt.savefig(history_dir + "loss_" + str(epoch).zfill(3) + "_plot.jpg")

            # -------

            acc_sum = 0
            for acc in epoch_acc_history_encoder:
                acc_sum += acc

            avg_acc = acc_sum / len(epoch_acc_history_encoder)
            avg_history_acc.append(avg_acc)

            with open(stats_acc, "a+") as f:
                f.write(str(avg_acc) + "\n")
                f.close()

            epoch_acc_history_encoder = []

            # -------

            # plt.clf()
            # plt.scatter(avg_index, avg_history_acc, s=20, label="Encoder accuracy")
            # plt.legend()
            # plt.show()
            # plt.savefig(history_dir + "acc_" + str(epoch).zfill(3) + "_plot.jpg")

            # -------

        if 0:

            test_avg_index.append(len(test_avg_index) + 1)

            _, ax = plt.subplots(4, 2, figsize=(16, 16))
            for i, fn in enumerate(test_images_A):
                test_image = cv2.imread(fn)
                test_image_tensor = numpy.expand_dims(test_image, 0)
                predict_image = autoencoder_B.predict(test_image_tensor)

                test_loss, test_acc = autoencoder_A.test_on_batch(predict_image, test_image_tensor)

                test_epoch_loss_history_encoder.append(test_loss)

                for i, fn in enumerate(test_images_A):
                    test_image = cv2.imread(fn)
                    test_image_tensor = numpy.expand_dims(test_image, 0)
                    predict_image = autoencoder_B.predict(test_image_tensor)

                    ax[i, 0].imshow(cv2.cvtColor(test_image_tensor[0], cv2.COLOR_BGR2RGB))
                    ax[i, 1].imshow(cv2.cvtColor(predict_image[0], cv2.COLOR_BGR2RGB))
                    ax[i, 0].set_title("Osoba A")
                    ax[i, 1].set_title("Osoba B")
                    ax[i, 0].axis("off")
                    ax[i, 1].axis("off")

            plt.clf()
            plt.show()
            plt.close()

            loss_sum = 0
            avg_loss = 0
            for loss in test_epoch_loss_history_encoder:
                loss_sum += loss

            avg_loss = loss_sum / len(test_epoch_loss_history_encoder)
            test_avg_history_loss.append(avg_loss)

            test_epoch_loss_history_encoder = []

            plt.clf()
            plt.scatter(test_avg_index, test_avg_history_loss, s=20, label="Encoder test")
            plt.legend()
            plt.show()
