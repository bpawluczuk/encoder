import datetime

import cv2
import numpy

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
    # x = downsample(128, strides=1)(x)
    x = downsample(128)(x)
    # x = downsample(512, strides=1)(x)
    x = downsample(256)(x)
    # x = downsample(512, strides=1)(x)
    x = downsample(512)(x)
    x = Flatten()(x)

    latent_space = Dense(_latent_dim)(x)

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)

    filters = 512
    x = upsampleShuffler(filters)(x)
    # x = upsampleShuffler(512, filter_times=4)(x)

    return Model(input_, x, name=name)


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 256))

    # filters = 512
    # filters //= 2
    # x = upsampleTranspose(filters)(input_)
    # x = downsample(filters, strides=1)(x)
    x = upsampleShuffler(256)(input_)

    # filters //= 2
    # x = upsampleTranspose(filters)(x)
    # x = downsample(filters, strides=1)(x)
    x = upsampleShuffler(128)(x)

    # filters //= 2
    # x = upsampleTranspose(filters)(x)
    # x = downsample(filters, strides=1)(x)
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

batch_size = 1
epochs = 100
dataset_size = len(images_A)
batches = round(dataset_size / batch_size)
plot_result_test = 1000
save_interval = 1000
sample_interval = 10

# ********************************************************************************

test_images_A = get_image_paths("data_train/OL_NEW/validOL")
test_images_B = get_image_paths("data_train/LU_NEW/validLU")

loss_history_A = []
loss_history_B = []
acc_history_A = []
acc_history_B = []
valid_history_A = []
valid_history_B = []

avg_index = []
avg_history_loss_A = []
avg_history_loss_B = []
avg_history_acc_A = []
avg_history_acc_B = []
avg_history_valid_A = []
avg_history_valid_B = []

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

        loss_history_A.append(loss_A[0])
        loss_history_B.append(loss_B[0])
        acc_history_A.append(loss_A[1])
        acc_history_B.append(loss_B[1])

        elapsed_time = datetime.datetime.now() - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [A loss: %f, acc: %3d%%] [B loss: %f, acc: %3d%%] time: %s " \
            % (epoch, epochs,
               batch, batches,
               loss_A[0], 100 * loss_A[1],
               loss_B[0], 100 * loss_B[1],
               elapsed_time))

        if batch % 2 == 0:

            avg_index.append(len(avg_index) + 1)

            # -------

            la_sum = 0
            for la in loss_history_A:
                la_sum += la

            avg_history_loss_A.append(la_sum / len(loss_history_A))

            loss_history_A = []

            # -------

            la_sum = 0
            for la in loss_history_B:
                la_sum += la

            avg_history_loss_B.append(la_sum / len(loss_history_B))

            loss_history_B = []

            plt.clf()
            plt.scatter(avg_index, avg_history_loss_A, s=30, label="Autoencoder A")
            plt.scatter(avg_index, avg_history_loss_B, s=30, label="Autoencoder B")
            plt.legend()
            plt.show()

            # -------

            la_sum = 0
            for la in acc_history_A:
                la_sum += la

            avg_history_acc_A.append((la_sum / len(acc_history_A)) * 100)

            acc_history_A = []

            # -------

            la_sum = 0
            for la in acc_history_B:
                la_sum += la

            avg_history_acc_B.append((la_sum / len(acc_history_B) * 100))

            acc_history_B = []

            plt.clf()
            plt.scatter(avg_index, avg_history_acc_A, s=30, label="Autoencoder A")
            plt.scatter(avg_index, avg_history_acc_B, s=30, label="Autoencoder B")
            plt.legend()
            plt.show()

            # score = autoencoder_A.evaluate(target_B, target_A, verbose=0)
            # print('Test loss:', score[0])
            # print('Test accuracy:', score[1])

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

        # if batch % plot_result_test == 0:
        #     image_test_A = get_image_paths("data_train/OL_TEST/trainTEST")
        #     ol = cv2.imread(image_test_A[0])
        #
        #     source_image_tensor_ol = numpy.expand_dims(ol, 0)
        #     predict_image_ol = autoencoder_B.predict(source_image_tensor_ol)[0]
        #     predict_image_ol = numpy.clip(predict_image_ol * 255, 0, 255).astype(numpy.uint8)
        #
        #     image_test_B = get_image_paths("data_train/OL_TEST/trainTEST")
        #     lu = cv2.imread(image_test_B[0])
        #
        #     source_image_tensor_lu = numpy.expand_dims(lu, 0)
        #     predict_image_lu = autoencoder_A.predict(source_image_tensor_lu)[0]
        #     predict_image_lu = numpy.clip(predict_image_lu * 255, 0, 255).astype(numpy.uint8)
        #
        #     _, ax = plt.subplots(2, 2, figsize=(12, 12))
        #     ax[0, 0].imshow(predict_image_ol)
        #     ax[0, 1].imshow(predict_image_lu)
        #     ax[0, 0].axis("off")
        #     ax[0, 1].axis("off")
        #
        #     plt.show()
        #     plt.close()
