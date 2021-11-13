import cv2

from lib.image_augmentation import random_transform, random_warp
from lib.utils import get_image_paths

# ********************************************************************

size = 256
zoom = 4  # 64*zoom

# ********************************************************************

images_A = get_image_paths("data/OL/trainOL")
images_B = get_image_paths("data/LU/trainLU")


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

inc = 0
for path in images_A:
    inc = inc + 1
    image = cv2.imread(path)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp(image, size, 5, zoom)
    cv2.imwrite("data_train/OL/trainOL/{i}_ol_warp.jpg".format(i=inc), target_img)
    cv2.imwrite("data_train/OL_WARP/trainOL/{i}_ol_warp.jpg".format(i=inc), warped_img)

inc = 0
for path in images_B:
    inc = inc + 1
    image = cv2.imread(path)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp(image, size, 5, zoom)
    cv2.imwrite("data_train/LU/trainLU/{i}_lu_warp.jpg".format(i=inc), target_img)
    cv2.imwrite("data_train/LU_WARP/trainLU/{i}_lu_warp.jpg".format(i=inc), warped_img)


# ********************************************************************
