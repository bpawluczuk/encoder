import os
import cv2

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame_512/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/OL_NEW/trainOL"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + "/" + str(inc) + "_ol.jpg", source_image)
