import cv2
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")


# Rozmiar nie jest określony, po pobraniu trzeba użyc funkcji resize

def getFaceAndCoordinates(source_image):

    source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    source_faces = frontal_face_detector(source_image_grayscale)

    for source_face in source_faces:
        source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
        source_face_landmark_points = []

    ymin = 10000
    xmin = 10000
    ymax = 0
    xmax = 0

    source_image_landmarks = source_image.copy()

    try:

        for landmark_no in range(0, 68):

            x_point = source_face_landmarks.part(landmark_no).x
            y_point = source_face_landmarks.part(landmark_no).y
            source_face_landmark_points.append((x_point, y_point))

            cv2.circle(source_image_landmarks, (x_point, y_point), 2, (255, 255, 0), -1)

            cv2.putText(
                source_image_landmarks,
                str(landmark_no),
                (x_point, y_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                .3,
                (255, 255, 255)
            )

            cv2.imshow("Landmark points", source_image_landmarks)

            # wyznaczenie punktow skrajnych

            if ymin > y_point:
                ymin = y_point
            if ymax < y_point:
                ymax = y_point
            if xmin > x_point:
                xmin = x_point
            if xmax < x_point:
                xmax = x_point

        # wyznaczenie szerokosci i wysokosci na podstawie punktow skrajnych
        h = ymax - ymin
        w = xmax - xmin

        color = (255, 0, 0)
        source_image_landmarks_rect = cv2.rectangle(source_image_landmarks, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.imshow("Landmark points - obrys", source_image_landmarks_rect)

        face = source_image[ymin:ymin + h, xmin:xmin + w]
        result = xmin, ymin, xmax, ymax, h, w, face

        # ************* Wyrownanie do idealnegon kwadratu ********************

        # srodek
        x0 = xmin + w // 2
        y0 = ymin + h // 2

        # wg szerokosci
        minXcorrect = x0 - w // 2
        minYcorrect = y0 - w // 2
        maxXcorrect = x0 + w // 2
        maxYcorrect = y0 + w // 2

        # poprawinie wysokosci i szerokosci
        h_correct = maxYcorrect - minYcorrect
        w_correct = maxXcorrect - minXcorrect

        face = source_image[minYcorrect:maxYcorrect, minXcorrect:maxXcorrect]

        color = (0, 255, 0)
        source_image_landmarks_rect = cv2.rectangle(source_image_landmarks,
                                                    (minXcorrect, minYcorrect),
                                                    (maxXcorrect, maxYcorrect), color, 1)
        cv2.imshow("Landmark points - korekta", source_image_landmarks_rect)

        result = minXcorrect, minYcorrect, maxXcorrect, maxYcorrect, h_correct, w_correct, face

        # *********************************

        return result

    except:
        print("Nie odczytano twarzy z obrazu")

    return None
