import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

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

height = 128
width = 128


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

input_img = Input(shape=(width, height, 1))

# encoder
# input = 28 x 28 x 1 (wide and thin)
conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
encoder_conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)

# decoder
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_conv3)  # 7 x 7 x 128
up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1

autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

# ********************************************************************

autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=4,
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
