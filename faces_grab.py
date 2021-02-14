import cv2
import numpy as np
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt

# from bing_image_downloader import downloader
# downloader.download(
#     "cloony",
#     limit=10000,
#     output_dir='dataset',
#     adult_filter_off=True,
#     force_replace=False,
#     timeout=60
# )


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source_dir = "C:\Sites\python\encoder\dataset\grab\cloony\images\\"
direct_dir = "C:\Sites\python\encoder\dataset\grab\cloony\faces\\"

for file in os.listdir(source_dir):
    image = cv2.imread(source_dir + file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier()
    faceCascade.load("haarcascade_frontalface_alt.xml")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite(direct_dir + str(w) + str(h) + '_faces.jpg', roi_color)


# inc = 0;
# for file in os.listdir('data/faces'):
#     inc = inc + 1
#     img = Image.open(file)
#     new_width = 256
#     new_height = 256
#     img = img.resize((new_width, new_height), Image.ANTIALIAS)
#     img.save('data/train/' + 'face_' + str(inc) + '.jpg')

#
# plt.figure()
# plt.imshow(image)
# plt.show()
