import numpy as np
import cv2
import pickle

width = 640
height = 480
threshold = 0.3
scaleVal = 1 + (400 / 1000)
nei = 9
bright = 120
area_temp = 16000
color = (0,225,255)
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
pickle_in = open("emotion_model.p", "rb")
model = pickle.load(pickle_in)
path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(path)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    success, img = cap.read()
    cap.set(10, 120)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleVal, nei)
    for (x,y,w,h) in objects:
        area = w*h
        if area > area_temp:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            roi_color = img[y:y+h, x:x+w]
            roi_color = np.asarray(roi_color)
            roi_color = cv2.resize(roi_color, (32, 32))
            roi_gray = preProcessing(roi_color)
            roi_gray = roi_gray.reshape(1, 32, 32, 1)
            classIndex = int(model.predict_classes(roi_gray))
            predictions = model.predict(roi_gray)
            probVal = np.amax(predictions)
            if classIndex == 0:
                name = "negative"
            else: name = "positive"
            if probVal > threshold:
                cv2.putText(img, str(name) + "  " + "{:0.2f}".format(probVal),
                            (x,y-5), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255), 1)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) == ord('q'):
        break