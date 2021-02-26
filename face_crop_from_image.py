import os
import cv2
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")

name = "bruce"
size = "128"

source_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_face_" + size + "/"

additional_size = 0


def getFace(source_image, rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    x_inc = int(w * additional_size)
    y_inc = int(h * additional_size)
    sub_face = source_image[y - y_inc:y + h + y_inc, x - x_inc:x + w + x_inc]
    if source_image is not None and sub_face.any():
        dest_image = cv2.resize(sub_face, (int(size), int(size)))
        cv2.imwrite(dest_dir + name + "_" + str(inc) + ".jpg", dest_image)


inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)
    if source_image is not None and source_image.any():
        faceRects = frontal_face_detector(source_image, 0)
        for faceRect in faceRects:
            getFace(source_image, faceRect)
