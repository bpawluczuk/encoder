import cv2
import mediapipe
import numpy as np

from lib.facial_landmarks import FaceLandmarks
from lib.seamless_image import seamless_images

face_landmarks = FaceLandmarks()
faceModule = mediapipe.solutions.face_mesh

oryginal_image_path = "output/GAN/laura_oliwka/img_5.jpg"
predicted_image_path = "output/GAN/laura_oliwka/predicted_img_5.jpg"

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

        result_image = cv2.add(background, predicted_face_extracted)

        seamless_destination_image = seamless_images(result_image, image)

        cv2.imshow('Oryginal image', image)
        cv2.imshow("Result image", seamless_destination_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
