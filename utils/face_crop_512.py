import os
import cv2
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("/Users/bpawluczuk/Sites/python/encoder/detect/shape_predictor_68_face_landmarks.dat")

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame_512/"

source_size = 512
dest_size = 256

def getFace(source_image, file_name, size, rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    x0 = x + w // 2
    y0 = y + h // 2

    x1 = x0 - 352
    y1 = y0 - 352
    x2 = x0 + 352
    y2 = y0 + 352

    sub_face = source_image[y1:y2, x1:x2]
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
            getFace(source_image, file_name, dest_size, faceRect)
