import os
import cv2
import mediapipe as mp

# source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/laura_frame/"
# dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/LU_NEW/trainLU/"

source_dir = "/Users/bpawluczuk/Sites/python/encoder/data/oliwia_frame/"
dest_dir = "/Users/bpawluczuk/Sites/python/encoder/data/OL_NEW/trainOL/"

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)


def getFace(source_image, file_name, size):
    img = source_image.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:

        for id, detection in enumerate(results.detections):

            mpDrawing.draw_detection(img, detection)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # cv2.rectangle(img, bbox, (255, 0, 255), 2)
            # cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
            #             (255, 0, 255), 2)

            sub_face = source_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            if source_image is not None and sub_face.any():
                sub_face = cv2.resize(sub_face, (int(size), int(size)))
                cv2.imwrite(dest_dir + file_name, sub_face)
                # cv2.imshow("sub_face", sub_face)

        # cv2.imshow("rect", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


source_size = 512
dest_size = 256

inc = 0
file_list = os.listdir(source_dir)
for file_name in sorted(file_list):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file_name)
    if source_image is not None and source_image.any():
        getFace(source_image, file_name, dest_size)
