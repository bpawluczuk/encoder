import cv2

dir = "/Users/bpawluczuk/Sites/python/encoder/data/"

vidcap = cv2.VideoCapture(dir + "lu_nowy.mp4")
vidcap.read()
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    print
    'Read a new frame: ', success
    if image is not None:
        cv2.imwrite(dir + "laura_frame/" + str(count).zfill(5) + ".jpg", image)  # save frame as JPEG file
        count += 1
