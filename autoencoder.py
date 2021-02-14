import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import sigmoid
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

# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\cloony\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\cloony\\"
#
# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\craig\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\craig\\"

height = 128
width = 128
chanels = 1


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

# ********************************************************************

input_img = Input(shape=(width, height, chanels), name='encoder_input')

# encoder
x = Conv2D(128, (3, 3), padding='same')(input_img)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), name='encoder_output', padding='same')(x)
x = LeakyReLU()(x)
encoder_output = x

encoder = Model(input_img, encoder_output, name="encoder_model")
encoder.summary()


# decoder
decoder_input = Input(shape=(32, 32, 64), name="decoder_input")

x = Conv2D(64, (3, 3), padding='same')(decoder_input)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), name='decoder_output', padding='same')(x)
x = sigmoid(x)
decoder_output = x

decoder = Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()

# ********************************************************************

autoencoder_input = Input(shape=(width, height, chanels), name="autoencoder_input")
autoencoder_encoder_output = encoder(autoencoder_input)

autoencoder_decoder_output = decoder(autoencoder_encoder_output)
autoencoder = Model(autoencoder_input, autoencoder_decoder_output, name="autoencoder")

autoencoder.summary()

autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

# ********************************************************************

autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=4,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoder.save('models/encoder_A.h5')
decoder.save('models/decoder_A.h5')
decoder.save('models/autoencoder_A.h5')

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
