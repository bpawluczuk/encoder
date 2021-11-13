import numpy

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# ********************************************************************

seed = 1
shuffle = False
dir = "OL_TEST"

validation_ol_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    # zoom_range=0.1,
    horizontal_flip=0.1
    # horizontal_flip=True
)

validation_ol = validation_ol_gen.flow_from_directory(
    'data_train/'+dir,
    class_mode=None,
    seed=seed,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=1,
    shuffle=shuffle,
)

validation_ol_dataset = tf.data.Dataset.from_generator(
    lambda: validation_ol,
    output_signature=(
        tf.TensorSpec(shape=(1, 256, 256, 3), dtype=tf.float32)
    ),
)

source_dir = "data_train/" + dir + "/trainTEST"
dest_dir = "data_train/" + dir + "/trainTEST"

print("validation_ol: " + str(len(validation_ol.filenames)))

plt.figure(figsize=(10, 10))
for i, image in enumerate(validation_ol_dataset.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.axis("off")
    plt.imsave(dest_dir + "/" + str(i + 1) + "_"+dir+".jpg", numpy.array(image[0], dtype=numpy.float32))

plt.show()
plt.close()
