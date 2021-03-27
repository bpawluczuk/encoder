import cv2

dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/frames/"


vidcap = cv2.VideoCapture(dir + "laura.mp4")
vidcap.read()
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    print
    'Read a new frame: ', success
    if image is not None:
        cv2.imwrite(dir+"laura/frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1
