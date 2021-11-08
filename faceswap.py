import cv2
import mediapipe as mp
import numpy as np
import dlib

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)
faceModule = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# oryginal_image_path = "data/laura_frame/00000.jpg"
# predicted_image_path = "output/AE/laura_oliwka/predicted_img_1.jpg"

oryginal_image_path = "data/oliwia_frame/00000.jpg"
predicted_image_path = "output/AE/oliwka_laura/predicted_img_1.jpg"

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

    return x, y, w, h, source_image, sub_face_resize


# --------- Oryginal ---------------

x, y, w, h, source_image, oryginal_image = getFaceCoordinates(cv2.imread(oryginal_image_path))

# oryginal_image = cv2.imread(oryginal_image_path)
cv2.imshow('oryginal_image', oryginal_image)
height, width, _ = oryginal_image.shape

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor(
    "/Users/bpawluczuk/Sites/python/encoder/detect/shape_predictor_68_face_landmarks.dat")

destination_image_grayscale = cv2.cvtColor(oryginal_image, cv2.COLOR_BGR2GRAY)
destination_faces = frontal_face_detector(destination_image_grayscale, 1)

for destination_face in destination_faces:

    destination_face_landmarks = frontal_face_predictor(destination_image_grayscale, destination_face)
    destination_face_landmark_points = []

    landmarks = []
    for landmark_no in range(0, 68):
        x_point = destination_face_landmarks.part(landmark_no).x
        y_point = destination_face_landmarks.part(landmark_no).y
        destination_face_landmark_points.append((x_point, y_point))

    landmarks = np.array(destination_face_landmark_points, np.int32)

    convexhull_oryginal = cv2.convexHull(landmarks)
    mask_oryginal = np.zeros((height, width), np.uint8)
    cv2.polylines(mask_oryginal, [convexhull_oryginal], True, 255, 3)

    image_oryginal_face_mask = cv2.fillConvexPoly(mask_oryginal, convexhull_oryginal, 255)
    image_oryginal_face_mask = cv2.bitwise_not(image_oryginal_face_mask)
    cv2.imshow('Oryginal image mask', image_oryginal_face_mask)

    background_oryginal = cv2.bitwise_and(oryginal_image, oryginal_image, mask=image_oryginal_face_mask)
    cv2.imshow('Oryginal image crop face', background_oryginal)

    image_oryginal_face = cv2.bitwise_and(oryginal_image, oryginal_image, mask=mask_oryginal)
    cv2.imshow('Oryginal image face', image_oryginal_face)

# --------- Predict ---------------

# predict_image = getFaceCoordinates(cv2.imread(predicted_image_path))
predict_image = cv2.imread(predicted_image_path)
cv2.imshow('image_predict', predict_image)

# ---------------------------------------------------------------

background_predict = cv2.bitwise_and(predict_image, predict_image, mask=image_oryginal_face_mask)
cv2.imshow('Predict image crop face', background_predict)

image_predict_face = cv2.bitwise_and(predict_image, predict_image, mask=mask_oryginal)
cv2.imshow('Predict image face', image_predict_face)

# ---------------------------------------------------------------

image_result = cv2.add(background_oryginal, image_predict_face)
cv2.imshow('Result', image_result)

# ------- Seamless --------------

(x, y, w, h) = (0, 0, 256, 256)
(x, y, w, h) = cv2.boundingRect(convexhull_oryginal)

center_x = (int((x + x + w) / 2))
center_y = (int((y + y + h) / 2))

center_face = (center_x, center_y)

result_image_seamless = cv2.seamlessClone(
    image_result,
    oryginal_image,
    mask_oryginal,
    center_face,
    cv2.NORMAL_CLONE
)

cv2.imshow('Result seamlessClone', result_image_seamless)

# ------- Result --------------

x, y, w, h, source_image, oryginal_image = getFaceCoordinates(cv2.imread(oryginal_image_path))

result_image_resize = cv2.resize(result_image_seamless, (w, h))

mask = np.zeros((h, w, 3), np.uint8)
source_image[y:(y + mask.shape[0]), x:(x + mask.shape[1])] = result_image_resize
cv2.imshow("image_oryginal_face_mask", source_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
