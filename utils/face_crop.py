import os
import cv2
import mediapipe as mp
import numpy as np

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/laura_frame/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/LU_NEW/trainLU/"

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/OL_NEW/trainOL/"

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
faceModule = mp.solutions.face_mesh


def getFace(source_image, file_name):
    img = source_image.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            sub_face = source_image[y:y + h, x:x + w]
            cv2.imshow("sub_face", sub_face)

            if source_image is not None and sub_face.any():
                sub_face = cv2.resize(sub_face, dest_size)
                cv2.imwrite(dest_dir + file_name, sub_face)
                # cv2.imshow("sub_face_resize", sub_face)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


inc = 0
file_list = os.listdir(source_dir)
for file_name in sorted(file_list):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file_name)
    if source_image is not None and source_image.any():
        getFace(source_image, file_name)
