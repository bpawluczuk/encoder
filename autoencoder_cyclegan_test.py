import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

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

run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

# *******************************************************************

IMAGE_SHAPE = (256, 256, 3)
image_size = (256, 256)

buffer_size = 1
batch_size = 1

# ********************************************************************

tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE

# ********************************************************************

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# ********************************************************************

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_train_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [*image_size])
    img = tf.image.random_crop(img, size=[*IMAGE_SHAPE])
    img = normalize_img(img)
    return img


def preprocess_test_image(img):
    img = tf.image.resize(img, [IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
    img = normalize_img(img)
    return img


# ********************************************************************

train_ol = tf.keras.preprocessing.image_dataset_from_directory(
    "data_test/OL",
    validation_split=0.2,
    subset="training",
    seed=1,
    label_mode=None,
    shuffle=buffer_size,
    image_size=image_size,
    batch_size=batch_size,
)

train_ol = (
    train_ol.map(normalize_img, num_parallel_calls=autotune).cache().shuffle(buffer_size)
)

train_lu = tf.keras.preprocessing.image_dataset_from_directory(
    "data_test/LU",
    validation_split=0.2,
    subset="training",
    seed=1,
    label_mode=None,
    shuffle=buffer_size,
    image_size=image_size,
    batch_size=batch_size,
)

train_lu = (
    train_lu.map(normalize_img, num_parallel_calls=autotune).shuffle(buffer_size)
)

# ********************************************************************

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_ol.take(4), train_lu.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(horse)
    ax[i, 1].imshow(zebra)
plt.show()


# *********************************************************************

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


# **********************************************************************

def residual_block(
        x,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=3,
        strides=1,
        padding="valid",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


# ********************************************************************************

def downsample(
        x,
        filters,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=4,
        strides=2,
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


# ********************************************************************************

def upsample(
        x,
        filters,
        activation,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=kernel_init,
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


# ********************************************************************************

def get_resnet_generator(
        filters=128,
        num_downsampling_blocks=2,
        num_residual_blocks=10,
        num_upsample_blocks=2,
        gamma_initializer=gamma_init,
        name=None,
):
    img_input = layers.Input(shape=IMAGE_SHAPE, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, 7, kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    # for _ in range(num_downsampling_blocks):
    #     filters *= 2
    x = downsample(x, filters=256, activation=layers.Activation("relu"), strides=2)
    x = downsample(x, filters=256, activation=layers.Activation("relu"), strides=2)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    # for _ in range(num_upsample_blocks):
    #     filters //= 2
    x = upsample(x, 256, activation=layers.Activation("relu"), strides=2)
    x = upsample(x, 256, activation=layers.Activation("relu"), strides=2)

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, 7, padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


# ********************************************************************************

def get_discriminator(filters=128, kernel_initializer=kernel_init, name=None):
    img_input = layers.Input(shape=IMAGE_SHAPE, name=name + "_img_input")
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_initializer)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=3, strides=2)
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=3, strides=1)

    x = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=kernel_initializer)(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# ********************************************************************************

class CycleGan(keras.Model):
    def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            lambda_cycle=10.0,
            lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer,
            gen_loss_fn,
            disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        cv2.waitKey(1)

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


# ********************************************************************************

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# ********************************************************************************

# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")

# ********************************************************************************

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)


# ********************************************************************************

class GANMonitor(keras.callbacks.Callback):

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_batch_begin(self, batch, logs=None):
        for i, img in enumerate(train_ol.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = ((prediction * 127.5) + 127.5).astype(np.uint8)
            img = ((img[0] * 127.5) + 127.5).numpy().astype(np.uint8)

            figure = np.stack((img, prediction))
            figure = np.concatenate(figure, axis=1)

            cv2.imshow("olTolu", figure)

        for i, img in enumerate(train_lu.take(self.num_img)):
            prediction = self.model.gen_F(img)[0].numpy()
            prediction = ((prediction * 127.5) + 127.5).astype(np.uint8)
            img = ((img[0] * 127.5) + 127.5).numpy().astype(np.uint8)

            figure = np.stack((img, prediction))
            figure = np.concatenate(figure, axis=1)

            cv2.imshow("luTool", figure)

        cv2.waitKey(1)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(train_ol.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

        plt.show()
        plt.close()

        cycle_gan_model.save_weights(("models/CycleGAN/cycleGan.h5"))


# ********************************************************************************

plotter = GANMonitor()

# ********************************************************************************

try:
    cycle_gan_model.built = True
    cycle_gan_model.load_weights("models/CycleGAN/cycleGan.h5")
    print("... load models")
except:
    print("models does not exist")

# ********************************************************************************

cycle_gan_model.fit(
    tf.data.Dataset.zip((train_ol, train_lu)),
    epochs=1,
    callbacks=[plotter],
)

# ********************************************************************************

cycle_gan_model.save_weights(("models/CycleGAN/cycleGan.h5"))

# ********************************************************************************

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, img in enumerate(train_ol.take(4)):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    # prediction = keras.preprocessing.image.array_to_img(prediction)
    # prediction.save("output/GAN/predicted_img_{i}.png".format(i=i))
plt.tight_layout()
plt.show()
