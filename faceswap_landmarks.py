import cv2
import mediapipe as mp
import numpy as np

from lib.facial_landmarks import FaceLandmarks
from lib.seamless_image import seamless_images

face_landmarks = FaceLandmarks()
faceModule = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

oryginal_image_path = "output/AE/laura_oliwka/img_1.jpg"
predicted_image_path = "output/AE/laura_oliwka/predicted_img_1.jpg"

dest_dir = "output/faceswap/AE"

with faceModule.FaceMesh(static_image_mode=True) as face:
    image = cv2.imread(predicted_image_path)
    height, width, _ = image.shape
    image_copy = image.copy()

    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    landmarks = []
    if results.multi_face_landmarks != None:
        landmarks = face_landmarks.get_facial_landmarks(image)

        convexhull = cv2.convexHull(landmarks)
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [convexhull], True, 255, 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        predicted_face_extracted = cv2.bitwise_and(image_copy, image_copy, mask=mask)

        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(image, image, mask=background_mask)

        result = cv2.add(background, predicted_face_extracted)

        cv2.imshow("Predicted image", image)
        cv2.imwrite(dest_dir + "/predicted_image.jpg", image)

        #################

        annotated_image = image.copy()

        for face_landmarks_list in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_list,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            cv2.imshow("Predicted image landmarks", annotated_image)
            cv2.imwrite(dest_dir + "/predicted_image_landmarks.jpg", annotated_image)

    ##########################################################

    with faceModule.FaceMesh(static_image_mode=True) as face:
        image = cv2.imread(oryginal_image_path)
        height, width, _ = image.shape
        frame_copy = image.copy()

        results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks = []
        if results.multi_face_landmarks != None:
            landmarks = face_landmarks.get_facial_landmarks(image)

            convexhull = cv2.convexHull(landmarks)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)

            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(image, image, mask=background_mask)
            cv2.imshow('Oryginal image mask', background_mask)
            cv2.imwrite(dest_dir + "/oryginal_image_mask.jpg", background_mask)

            result_image = cv2.add(background, predicted_face_extracted)
            cv2.imshow('Oryginal image extract face', predicted_face_extracted)
            cv2.imwrite(dest_dir + "/predicted_face_extracted.jpg", predicted_face_extracted)

            seamless_destination_image = seamless_images(result_image, image)

            cv2.imshow('Oryginal image', image)
            cv2.imshow("Result image", seamless_destination_image)

            cv2.imwrite(dest_dir + "/result_image.jpg", seamless_destination_image)
            cv2.imwrite(dest_dir + "/oryginal_image.jpg", image)

        #################

        annotated_image = image.copy()

        for face_landmarks_list in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_list,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            cv2.imshow("Oryginal image landmarks", annotated_image)
            cv2.imwrite(dest_dir + "/oryginal_image_landmarks.jpg", annotated_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
