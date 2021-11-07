import os
import cv2
import mediapipe as mp
import numpy as np

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/laura_frame/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/LU_NEW/trainLU/"

# source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame/"
# dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/OL_NEW/trainOL/"

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
faceModule = mp.solutions.face_mesh


def getFaceCoordinates(source_image, additional_size_from_center=216):
    with faceModule.FaceMesh(static_image_mode=True) as face:
        img = source_image.copy()
        height, width, _ = img.shape

        results = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        facelandmarks = []
        for facial_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])

            facelandmarks = np.array(facelandmarks, np.int32)

            if facelandmarks.any():
                convexhull = cv2.convexHull(facelandmarks)
                mask = np.zeros((height, width), np.uint8)

        (x, y, w, h) = cv2.boundingRect(convexhull)
        cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 2)
        cv2.imshow('convexhull box', img)

        center_x = (int((x + x + w) / 2))
        center_y = (int((y + y + h) / 2))

        x1 = center_x - additional_size_from_center
        x2 = center_x + additional_size_from_center
        y1 = center_y - additional_size_from_center
        y2 = center_y + additional_size_from_center

        w = x2 - x1
        h = y2 - y1

        cv2.rectangle(img, (x1, y1, w, h), (255, 127, 127), 3)
        cv2.imshow('Result box', img)

        sub_face = source_image[y1:y2, x1:x2]
        center_face = (additional_size_from_center, additional_size_from_center)
        result_image = cv2.seamlessClone(
            sub_face,
            sub_face,
            mask,
            center_face,
            cv2.NORMAL_CLONE
        )
        # cv2.imshow('seamlessClone', result_image)

    return x1, x2, y1, y2, w, h


def getFace(source_image, file_name):
    img = source_image.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)

    dest_size = 256

    if results.detections:

        for id, detection in enumerate(results.detections):

            mpDrawing.draw_detection(img, detection)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

            x1, x2, y1, y2, w, h = getFaceCoordinates(img, 216)

            sub_face = source_image[y1:y2, x1:x2]

            if source_image is not None and sub_face.any():
                sub_face = cv2.resize(sub_face, (int(w), int(h)))
                # cv2.imwrite(dest_dir + file_name, sub_face)
                # cv2.imshow("sub_face", sub_face)

                sub_face_resize = cv2.resize(sub_face, (int(dest_size), int(dest_size)))
                cv2.imwrite(dest_dir + file_name, sub_face_resize)
                # cv2.imshow("sub_face_resize", sub_face_resize)

    # with faceModule.FaceMesh(static_image_mode=True) as face:
    #     image = sub_face.copy()
    #     height, width, _ = image.shape
    #
    #     results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # for face_landmarks_list in results.multi_face_landmarks:
    #     mpDrawing.draw_landmarks(
    #         image=image,
    #         landmark_list=face_landmarks_list,
    #         connections=mp_face_mesh.FACEMESH_TESSELATION,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    #
    #     cv2.imshow("Predicted image landmarks", image)
    #

    # cv2.imshow("rect", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


inc = 0
file_list = os.listdir(source_dir)
for file_name in sorted(file_list):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file_name)
    if source_image is not None and source_image.any():
        getFace(source_image, file_name)
