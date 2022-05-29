import pickle
import cv2
import os
import numpy as np
import imutils
import time
from threading import Thread
import playsound
import argparse
import telebot
tb = telebot.TeleBot("1777923430:AAHXysR9_eow-FTlyNiVtdT0q5nr__lxuFQ")
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
args = vars(ap.parse_args())


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False
ALERTS_SENT = False
BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
modelPath = os.path.join(
    BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
embedding_model = os.path.join(BASE_DIR, 'nn4.small2.v1.t7')
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)
# load the actual face recognition model along with the label encoder
recognizer_file = os.path.join(BASE_DIR, 'output/recognizer.pickle')
le_file = os.path.join(BASE_DIR, 'output/le.pickle')
recognizer = pickle.loads(open(recognizer_file, "rb").read())
le = pickle.loads(open(le_file, "rb").read())
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.65:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0),
                                             swapRB=True,
                                             crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            # draw the bounding box of the face along with the
            # associated probability
            if proba * 100 > 60 and name != 'unknown':
                text = "{}: {:.2f}%".format(name, proba * 100)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                COUNTER = 0
                ALARM_ON = False
                ALERTS_SENT = False
            else:
                text = "{}: {:.2f}%".format('unknowm', proba * 100)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                        # draw an alarm on the frame
                    cv2.putText(frame, "Unkown ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not ALERTS_SENT:
                        cv2.imwrite('unknown.jpg', frame)
                        foto = open('unknown.jpg', 'rb')
                        tb.send_photo('1298675596', foto)
                        print('Unkown person detected. sending alerts....')
                        ALERTS_SENT = True
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
