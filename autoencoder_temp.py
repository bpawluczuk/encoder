import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from umeyama import umeyama

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Reshape
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

test_dir = "/Users/bpawluczuk/Sites/python/VAE/data/test/clooney/"
train_dir = "/Users/bpawluczuk/Sites/python/VAE/data/train/clooney/"

# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\cloony\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\cloony\\"

# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\craig\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\craig\\"

jason_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/jason/"
jasonfake_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/jasonfake/"

height = 256
width = 256
chanels = 1

IMAGE_SHAPE = (256, 256, 3)
ENCODER_DIM = 1024

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)


# ********************************************************************

def imagetensor(imagedir, width, height):
    for i, im in enumerate(os.listdir(imagedir)):
        image = Image.open(imagedir + im)
        # image = image.convert('L')
        image = image.resize((width, height), Image.ANTIALIAS)
        if i == 0:
            images = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
        else:
            image = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
            images = np.append(images, image, axis=0)

    return images


x_train = imagetensor(train_dir, width, height)
x_test = x_train


# ********************************************************************

def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def upscale(filters):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        return x

    return block


# ********************************************************************

# encoder

def Encoder():
    input_ = Input(shape=IMAGE_SHAPE, name='encoder_input')
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Dense(ENCODER_DIM)(Flatten()(x))
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale(128)(x)
    return Model(input_, x, name="encoder_model")


# decoder

def Decoder():
    input_ = Input(shape=(8, 8, 512), name="decoder_input")
    x = input_
    x = upscale(1024)(x)
    x = upscale(512)(x)
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    print(x.shape)
    return Model(input_, x, name="decoder_model")


# ********************************************************************

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

autoencoder_A.summary()

x_train = imagetensor(jasonfake_dir, width, height)
x_test = imagetensor(jasonfake_dir, width, height)

autoencoder_A.fit(x_train, x_train,
                  epochs=10,
                  batch_size=4,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# ********************************************************************

# autoencoder.fit(x_train, x_train,
#                 epochs=200,
#                 batch_size=4,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# encoder.save('models/encoder_B.h5')
# decoder.save('models/decoder_B.h5')
# decoder.save('models/autoencoder_B.h5')

# ********************************************************************

decoded_imgs = autoencoder_A.predict(x_test)
cv2.imshow("", decoded_imgs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


# ********************************************************************

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


# get pair of random warped images from aligened face image
def random_warp(image):
    shape = image.shape
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)

    interp_mapx = np.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = np.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    cv2.imshow("get_training_data 2", warped_image)

    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64, 64))

    return warped_image, target_image


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        # cv2.imshow("get_training_data 1", image)
        image = random_transform(image, **random_transform_args)
        # cv2.imshow("get_training_data 2", image)
        warped_img, target_img = random_warp(image)
        # cv2.imshow("get_training_data 3", target_img)
        # cv2.imshow("get_training_data 4", warped_img)

        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images


image1 = cv2.imread("/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/jason.jpg")
image1 = cv2.resize(image1, (width, height), interpolation=cv2.INTER_AREA)
image1 = image1 / 255.0

image2 = cv2.imread("/Users/bpawluczuk/Sites/python/encoder/dataset/deepfake/brucewills.jpg")
image2 = cv2.resize(image2, (width, height), interpolation=cv2.INTER_AREA)
image2 = image2 / 255.0

# wraped_A, target_A = get_training_data([image1, image2], 1)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
