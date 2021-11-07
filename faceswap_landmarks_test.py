import cv2
import mediapipe as mp
import numpy as np

faceModule = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

oryginal_image_path = "output/AE/laura_oliwka/img_1.jpg"
predicted_image_path = "output/AE/laura_oliwka/predicted_img_1.jpg"

# oryginal_image_path = "output/AE/oliwka_laura/img_1.jpg"
# predicted_image_path = "output/AE/oliwka_laura/predicted_img_1.jpg"

dest_dir = "output/faceswap/AE"


def getFaceCoordinates(source_image, additional_size_from_center=192):
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

        sub_face = img[y1:y2, x1:x2]
        center_face = (additional_size_from_center, additional_size_from_center)
        result_image = cv2.seamlessClone(
            sub_face,
            sub_face,
            mask,
            center_face,
            cv2.NORMAL_CLONE
        )
        cv2.imshow('seamlessClone', result_image)

        cv2.waitKey(0)

    return x1, x2, y1, y2, w, h

with faceModule.FaceMesh(static_image_mode=True) as face:
    image_predict = cv2.imread(predicted_image_path)
    height, width, _ = image_predict.shape

    results = face.process(cv2.cvtColor(image_predict, cv2.COLOR_BGR2RGB))

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
        cv2.fillConvexPoly(mask_predict, convexhull_predict, 255)

        background_mask_predict = cv2.bitwise_not(mask_predict)
        background = cv2.bitwise_and(image_predict, image_predict, mask=background_mask_predict)

        cv2.imshow("Predicted image mask", background_mask_predict)

        predicted_face_extracted = cv2.bitwise_and(image_predict, image_predict, mask=mask_predict)
        cv2.imshow("Predicted face", predicted_face_extracted)

        cv2.imshow("Predicted image", image_predict)
        cv2.imwrite(dest_dir + "/predicted_image.jpg", image_predict)

        #################

        annotated_image = image_predict.copy()

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
    image_oryginal = cv2.imread(oryginal_image_path)
    height, width, _ = image_oryginal.shape

    # x1, x2, y1, y2, w, h = getFaceCoordinates(image_oryginal, additional_size_from_center=216)

    results = face.process(cv2.cvtColor(image_oryginal, cv2.COLOR_BGR2RGB))

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

            final_destination_face_mask = cv2.fillConvexPoly(mask_oryginal, convexhull_oryginal, 255)
            final_destination_face_mask = cv2.bitwise_not(final_destination_face_mask)
            cv2.imshow('final_destination_face_mask', final_destination_face_mask)

            background_oryginal = cv2.bitwise_and(image_oryginal, image_oryginal, mask=final_destination_face_mask)
            cv2.imshow('Oryginal image mask', background_oryginal)
            # cv2.imwrite(dest_dir + "/oryginal_image_mask.jpg", background_mask)

            image_result = cv2.add(background_oryginal, predicted_face_extracted)
            cv2.imshow('Result', image_result)
            # cv2.imwrite(dest_dir + "/predicted_face_extracted.jpg", predicted_face_extracted)

            (x, y, w, h) = cv2.boundingRect(convexhull_oryginal)
            # cv2.rectangle(image_oryginal, (x, y, w, h), (255, 0, 255), 2)
            # cv2.imshow('Result2', image_oryginal)

            center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
            result_image = cv2.seamlessClone(
                image_result,
                image_oryginal,
                mask_oryginal,
                center_face,
                cv2.NORMAL_CLONE
            )

            cv2.imshow('Result seamlessClone', result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()