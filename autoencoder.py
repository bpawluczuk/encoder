import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************


train_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/bruce/"

bruce_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/bruce/"
jason_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/jason/"
jasonfake_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/jasonfake/"
cloony_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/cloony/"

train_cloony = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/cloony/"
train_craig = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/craig/"

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

height = 256
width = 256
chanels = 1

IMAGE_SHAPE = (256, 256, 1)
ENCODER_DIM = 1024


# ********************************************************************

def imagetensor(imagedir, width, height):
    for i, im in enumerate(os.listdir(imagedir)):
        image = Image.open(imagedir + im)
        image = image.convert('L')
        image = image.resize((width, height), Image.ANTIALIAS)
        if i == 0:
            images = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
        else:
            image = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
            images = np.append(images, image, axis=0)

    return images


# ********************************************************************

def Encoder():
    input_ = Input(shape=(width, height, chanels), name='encoder_input')
    x = input_
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), name='encoder_output', padding='same')(x)
    x = LeakyReLU()(x)
    return Model(input_, x, name="encoder_model")


def Decoder():
    input_ = Input(shape=(64, 64, 64), name='encoder_input')
    x = input_
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), name='decoder_output', padding='same')(x)
    x = sigmoid(x)
    return Model(input_, x, name="decoder_model")


# ********************************************************************

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizers.RMSprop(), loss='mean_squared_error')
autoencoder_B.compile(optimizer=optimizers.RMSprop(), loss='mean_squared_error')

# ********************************************************************

# x_train_1 = imagetensor(train_cloony, width, height)
# x_test_1 = x_train_1

x_train_1 = imagetensor(jasonfake_dir, width, height)
x_test_1 = imagetensor(jasonfake_dir, width, height)

autoencoder_B.fit(x_train_1, x_train_1,
                  epochs=10,
                  batch_size=4,
                  shuffle=True,
                  validation_data=(x_test_1, x_test_1),
                  callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# x_train_2 = imagetensor(train_craig, width, height)
# x_test_2 = x_train_2
#
# autoencoder_B.fit(x_train_2, x_train_2,
#                   epochs=20,
#                   batch_size=4,
#                   shuffle=True,
#                   validation_data=(x_test_2, x_test_2),
#                   callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# encoder.save_weights("models/encoder.h5")
# decoder_A.save_weights("models/decoder_A.h5")
# decoder_B.save_weights("models/decoder_B.h5")

# try:
#     encoder.load_weights("models/encoder.h5")
#     decoder_A.load_weights("models/decoder_A.h5")
#     decoder_B.load_weights("models/decoder_B.h5")
# except:
#     pass

# decoded_bruce = autoencoder_A.predict(x_test_1)
# cv2.imshow("bruce", decoded_bruce[0])
#
# decoded_jason = autoencoder_B.predict(x_test_2)
# cv2.imshow("jason", decoded_jason[0])

x_train_3 = imagetensor(jason_dir, width, height)
x_test_3 = x_train_3

decoded_result = autoencoder_B.predict(x_test_3)
cv2.imshow("result", decoded_result[0])

cv2.waitKey(0)
cv2.destroyAllWindows()

# n = 5
# plt.figure(figsize=(4, 4))
# for i in range(1, n + 1):
#     # Display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(width, height))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(width, height))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# decoded_img = autoencoder.predict(x_test)
# plt.figure()
# plt.gray()
# plt.imshow(decoded_img.reshape(width, height))
# plt.show()
