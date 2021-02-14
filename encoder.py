import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from keras.callbacks import TensorBoard
from PIL import Image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************

test_dir = "/Users/bpawluczuk/Sites/python/VAE/data/test/clooney/"
train_dir = "/Users/bpawluczuk/Sites/python/VAE/data/train/clooney/"

train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\cloony\\"
test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\cloony\\"

height = 256
width = 256


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


x_train = imagetensor(train_dir, width, height)
x_test = x_train

# x_test = imagetensor(test_dir, width, height)

# ********************************************************************

input_img = keras.Input(shape=(width, height, 1))

# encoder
# input = 28 x 28 x 1 (wide and thin)
conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
conv2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
conv3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)

# decoder
conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
up1 = layers.UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
up2 = layers.UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

# ********************************************************************

autoencoder.fit(x_train, x_train,
                epochs=300,
                batch_size=20,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder.save('encoder_1.h5')

# autoencoder = load_model('encoder_1.h5')

decoded_imgs = autoencoder.predict(x_test)

n = 5
plt.figure(figsize=(4, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# decoded_img = autoencoder.predict(x_test)
# plt.figure()
# plt.gray()
# plt.imshow(decoded_img.reshape(width, height))
# plt.show()
