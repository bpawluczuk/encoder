import os
import cv2

name = "laura"
source_dir = "dataset/frames/laura_512/"
dest_dir = "dataset/frames/laura_128/"
size = 128

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        dest_image = cv2.resize(source_image, (int(size), int(size)))
        cv2.imwrite(dest_dir + name + "_128_" + str(inc) + ".jpg", dest_image)
