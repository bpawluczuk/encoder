import os
import cv2
import numpy as np
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor(
    "/Users/bpawluczuk/Sites/python/encoder/detect/shape_predictor_68_face_landmarks.dat")

name = "bruce"
source_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_face/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_face_mask/"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + name + "_face_mask_" + str(inc) + ".jpg", source_image)
