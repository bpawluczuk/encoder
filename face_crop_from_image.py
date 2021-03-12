import os
import cv2
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")

source_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/rowan/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/rowan_512/"

additional_size = 0


def getFace(source_image, file_name, size, rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    x_inc = int(w * additional_size)
    y_inc = int(h * additional_size)

    x1 = x - x_inc
    y1 = y - y_inc
    x2 = x + w + x_inc
    y2 = y + h + y_inc

    x0 = x + w // 2
    y0 = y + h // 2

    x11 = x0 - 256
    y11 = y0 - 256
    x22 = x0 + 256
    y22 = y0 + 256

    sub_face = source_image[y11:y22, x11:x22]
    if source_image is not None and sub_face.any():
        sub_face = cv2.resize(sub_face, (int(size), int(size)))
        cv2.imwrite(dest_dir + file_name, sub_face)


inc = 0
for file_name in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file_name)
    if source_image is not None and source_image.any():
        faceRects = frontal_face_detector(source_image, 0)
        for faceRect in faceRects:
            getFace(source_image, file_name, 512, faceRect)
