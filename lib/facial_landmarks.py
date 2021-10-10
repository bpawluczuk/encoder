import mediapipe as mp
import cv2
import numpy as np


class FaceLandmarks:
    def __init__(self):

        self.face_module = mp.solutions.face_mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.drawingModule = mp.solutions.drawing_utils

    def get_facial_landmarks(self, image):

        height, width, _ = image.shape
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        # circleDrawingSpec = self.drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        # lineDrawingSpec = self.drawingModule.DrawingSpec(thickness=1, color=(0, 255, 0))

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            # self.drawingModule.draw_landmarks(
            #     image, facial_landmarks,
            #     self.face_module.FACEMESH_FACE_OVAL,
            #     circleDrawingSpec, lineDrawingSpec)
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])

        return np.array(facelandmarks, np.int32)
