import os
import cv2

source_dir = "/data_train/LU/trainLU"
dest_dir = "/data_train/LU_GAN/trainLU"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + "/" + str(inc) + "_lu.jpg", source_image)
