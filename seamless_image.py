import cv2
import numpy as np
import dlib


def seamless_images(source_image_path, destination_image_path):
    frontal_face_detector = dlib.get_frontal_face_detector()
    frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")

    source_image = cv2.imread(source_image_path)

    destination_image = cv2.imread(destination_image_path)
    destination_image_grayscale = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)

    destination_faces = frontal_face_detector(destination_image_grayscale)

    for destination_face in destination_faces:

        destination_face_landmarks = frontal_face_predictor(destination_image_grayscale, destination_face)
        destination_face_landmark_points = []

        for landmark_no in range(0, 68):
            x_point = destination_face_landmarks.part(landmark_no).x
            y_point = destination_face_landmarks.part(landmark_no).y
            destination_face_landmark_points.append((x_point, y_point))

        destination_face_landmark_points_array = np.array(destination_face_landmark_points, np.int32)
        destination_face_convexhull = cv2.convexHull(destination_face_landmark_points_array)

    final_destination_canvas = np.zeros_like(destination_image_grayscale)
    final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destination_face_convexhull, 255)

    (x, y, w, h) = cv2.boundingRect(destination_face_convexhull)
    destination_face_center_point = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlesscloned_face = cv2.seamlessClone(
        source_image,
        destination_image,
        final_destination_face_mask,
        destination_face_center_point,
        cv2.NORMAL_CLONE
    )

    output_file = "seamless/final_destination_face_mask.jpg"
    cv2.imwrite(str(output_file), final_destination_face_mask)

    return seamlesscloned_face