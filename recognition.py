from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
from build_face import build_face
from extract_embeddings import extract_embeddings
from train_model import train_model

print("[INFO] loading face detector")
protoPath = os.path.sep.join(["models", "deploy.prototxt"])
modelPath = os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer")
embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

id_index = 1
count = 0

for _ in range(15):
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			if name == "unknown":
				count += 1
			else:
				count -= 1

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	fps.update()
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	time.sleep(0.25)


def newdir(id_index):
	path = 'data/id%d' % id_index
	if not os.path.exists(path):
		os.mkdir(path, mode=0o777)
		build_face(path)
	else:
		id_index += 1
		newdir(id_index)


if count > 0:
	newdir(id_index)
	extract_embeddings()
	train_model()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()