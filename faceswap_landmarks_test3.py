import cv2
import mediapipe as mp
import numpy as np

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)
faceModule = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# oryginal_image_path = "output/AE/laura_oliwka/img_1.jpg"
# predicted_image_path = "output/AE/laura_oliwka/predicted_img_1.jpg"

oryginal_image_path = "data/oliwia_frame/00000.jpg"
predicted_image_path = "data/laura_frame/00000.jpg"

oryginal_image_path = "data/laura_frame/00000.jpg"
predicted_image_path = "data/oliwia_frame/00000.jpg"

dest_dir = "output/faceswap/AE"


def getFaceCoordinates(source_image):
    img = source_image.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img)

    results = faceDetection.process(imgRGB)

    dest_size = (256, 256)

    if results.detections:

        for id, detection in enumerate(results.detections):
            mpDrawing.draw_detection(img, detection)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

            cv2.imshow("bboxC", img)

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            sub_face = source_image[y:y + h, x:x + w]
            # cv2.imshow("sub_face", sub_face)

            sub_face_resize = cv2.resize(sub_face, dest_size)
            # cv2.imshow("sub_face_resize", sub_face_resize)

    return sub_face_resize


#--------- Predict ---------------

predict_image = getFaceCoordinates(cv2.imread(predicted_image_path))
cv2.imshow('image_predict', predict_image)

with faceModule.FaceMesh(static_image_mode=True) as face:
    image = predict_image.copy()
    height, width, _ = image.shape

    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    facelandmarks = []
    for facial_landmarks in results.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            facelandmarks.append([x, y])

        facelandmarks = np.array(facelandmarks, np.int32)

        if facelandmarks.any():
            convexhull_predict = cv2.convexHull(facelandmarks)
            mask_predict = np.zeros((height, width), np.uint8)
            cv2.polylines(mask_predict, [convexhull_predict], True, 255, 3)

            # ------- Predict Image -------------

            image_predict_face_mask = cv2.fillConvexPoly(mask_predict, convexhull_predict, 255)
            image_predict_face_mask = cv2.bitwise_not(image_predict_face_mask)
            cv2.imshow('Predict image mask', image_predict_face_mask)

            background_predict = cv2.bitwise_and(image, image, mask=image_predict_face_mask)
            cv2.imshow('Predict image crop face', background_predict)

            image_predict_face = cv2.bitwise_and(image, image, mask=mask_predict)
            cv2.imshow('Predict image face', image_predict_face)


#--------- Oryginal ---------------


oryginal_image = getFaceCoordinates(cv2.imread(oryginal_image_path))
cv2.imshow('image_oryginal', oryginal_image)

with faceModule.FaceMesh(static_image_mode=True) as face:
    image = oryginal_image.copy()
    height, width, _ = image.shape

    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    facelandmarks = []
    for facial_landmarks in results.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            facelandmarks.append([x, y])

        facelandmarks = np.array(facelandmarks, np.int32)

        if facelandmarks.any():
            convexhull_oryginal = cv2.convexHull(facelandmarks)
            mask_oryginal = np.zeros((height, width), np.uint8)
            cv2.polylines(mask_oryginal, [convexhull_oryginal], True, 255, 3)

            # ------- Oryginal Image -------------

            image_oryginal_face_mask = cv2.fillConvexPoly(mask_oryginal, convexhull_oryginal, 255)
            image_oryginal_face_mask = cv2.bitwise_not(image_oryginal_face_mask)
            cv2.imshow('Oryginal image mask', image_oryginal_face_mask)

            background_oryginal = cv2.bitwise_and(image, image, mask=image_oryginal_face_mask)
            cv2.imshow('Oryginal image crop face', background_oryginal)

            image_oryginal_face = cv2.bitwise_and(image, image, mask=mask_oryginal)
            cv2.imshow('Oryginal image face', image_oryginal_face)





            # image_result = cv2.add(background_predict, image_predict_face)
            # cv2.imshow('Result', image_result)



    # (x, y, w, h) = cv2.boundingRect(convexhull_oryginal)
    #
    # center_x = (int((x + x + w) / 2))
    # center_y = (int((y + y + h) / 2))
    #
    # center_face = (center_x, center_y)
    #
    # result_image = cv2.seamlessClone(
    #             image_oryginal,
    #             image_result,
    #             mask_oryginal_zeros,
    #             center_face,
    #             cv2.NORMAL_CLONE
    #         )
    #
    # cv2.imshow('Result seamlessClone', result_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
