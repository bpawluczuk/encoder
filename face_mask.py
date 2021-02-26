import os
import cv2
import numpy as np
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor(
    "/Users/bpawluczuk/Sites/python/encoder/detect/shape_predictor_68_face_landmarks.dat")

name = "matt"
source_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_256/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "/"
dest_dir_canvas = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_canvas/"
dest_dir_face = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/" + name + "_face/"

inc = 0
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():

        cv2.imwrite(dest_dir + name + "_" + str(inc) + ".jpg", source_image)
        source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        source_image_canvas = np.zeros_like(source_image_grayscale)
        source_faces = frontal_face_detector(source_image_grayscale)

        for source_face in source_faces:

            source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
            source_face_landmark_points = []

            for landmark_no in range(0, 68):
                x_point = source_face_landmarks.part(landmark_no).x
                y_point = source_face_landmarks.part(landmark_no).y
                source_face_landmark_points.append((x_point, y_point))

            source_face_landmark_points_array = np.array(source_face_landmark_points, np.int32)

            source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)
            cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)
            cv2.imwrite(dest_dir_canvas + name + "_face_canvas_" + str(inc) + ".jpg", source_image_canvas)

            # place the created mask over the source image
            source_face_image = cv2.bitwise_and(source_image, source_image, mask=source_image_canvas)
            cv2.imwrite(dest_dir_face + name + "_face_" + str(inc) + ".jpg", source_face_image)
