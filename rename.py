import os
import cv2

source_dir = "data_train/OL_WARP/trainOL"
dest_dir = "data_train/OL_GAN/trainOL"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + "/" + file)

    if source_image is not None and source_image.any():
        # cv2.imwrite(dest_dir + "/" + str(inc) + "_lu.jpg", source_image)
        cv2.imwrite(dest_dir + "/" + str(inc) + "_" + str(inc) + "_lu.jpg", source_image)
