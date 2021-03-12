import cv2
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")


def getFace(source_image, size):
    faceRects = frontal_face_detector(source_image, 0)
    for faceRect in faceRects:
        x = faceRect.left()
        y = faceRect.top()
        w = faceRect.right() - x
        h = faceRect.bottom() - y

        offset = 40
        sub_face = source_image[x - offset:x + w + offset, y - offset:y + h + offset]

        sub_face = cv2.resize(sub_face, (int(size), int(size)))
    return sub_face
