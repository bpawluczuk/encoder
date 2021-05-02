import os
import cv2
import numpy as np
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor(
    "/Users/bpawluczuk/Sites/python/encoder/detect/shape_predictor_68_face_landmarks.dat")

source_dir = "/Users/bpawluczuk/Sites/python/dataset/frames/laura_256/laura_256/"
dest_dir = "/Users/bpawluczuk/Sites/python/dataset/frames/trainLU/"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + str(inc) + "_lu.jpg", source_image)
