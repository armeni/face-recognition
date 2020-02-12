from imutils.video import VideoStream
import imutils
import time
import cv2
import os


def build_face(path):
	detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
	vs = VideoStream(src=0 + cv2.CAP_DSHOW).start()

	time.sleep(2.0)
	total = 0

	for _ in range(15):
		frame = vs.read()
		orig = frame.copy()
		frame = imutils.resize(frame, width=400)

		rects = detector.detectMultiScale(
			cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
			minNeighbors=5, minSize=(30, 30))

		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

		time.sleep(0.25)

	cv2.destroyAllWindows()
	vs.stop()