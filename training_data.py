import numpy
from image_augmentation import random_transform
from image_augmentation import random_warp, random_warp2
import cv2

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def get_training_data(images, batch_size):
    indices = numpy.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        cv2.imshow("one", image)
        image = random_transform(image, **random_transform_args)
        cv2.imshow("two", image)
        warped_img, target_img = random_warp2(image)
        # warped_img, target_img = random_warp(image)

        # cv2.imshow("warped_image", warped_img)
        # cv2.imshow("target_img", target_img)

        if i == 0:
            warped_images = numpy.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = numpy.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    # key = cv2.waitKey(0)
    return warped_images, target_images
