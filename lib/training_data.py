import numpy
from lib.image_augmentation import random_transform
from lib.image_augmentation import random_warp
import cv2
from lib.utils import stack_images
import matplotlib.pyplot as plt

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def get_training_data(images, batch_size, size=128, zoom=2):

    indices = numpy.random.randint(len(images), size=batch_size)

    for i, index in enumerate(indices):

        # print("Image index: "+str(index))
        image = images[index]

        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image, size=size, scale=5, zoom=zoom)

        # cv2.imshow("warped_image", warped_img)
        # cv2.imshow("target_img", target_img)



        # _, ax = plt.subplots(4, 4, figsize=(20, 20))
        # ax[i, 0].imshow(warped_img)
        # ax[i, 1].imshow(target_img)
        # ax[i, 0].set_title("Osoba A")
        # ax[i, 1].set_title("Osoba B")
        # ax[i, 0].axis("off")
        # ax[i, 1].axis("off")
        # plt.show()
        # plt.close()



        if i == 0:
            warped_images = numpy.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = numpy.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    #     figure = numpy.stack([
    #         warped_img,
    #         target_img,
    #     ], axis=1)
    #     figure = numpy.concatenate([figure], axis=0)
    #     figure = stack_images(figure)
    #     figure = numpy.clip(figure * 255, 0, 255).astype(numpy.uint8)
    #     cv2.imshow("Results Augmentation", figure)
    #
    key = cv2.waitKey(0)
    # print(len(warped_images))
    # print(len(target_images))
    # print("*********")
    return warped_images.astype('float32'), target_images.astype('float32')
