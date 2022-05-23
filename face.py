import numpy as np
import cv2

# Loading the required haar-cascade xml classifier file
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye = cv2.CascadeClassifier('haarcascade_eye.xml')

# Reading the image
img = cv2.imread('hello2.jpg')
# Converting image to grayscale because initiall image in three layers i.e. rgb so convert into one layer
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying the face detection method on the grayscale image
faces = face.detectMultiScale(gray, 1.1, 5)
print(faces)

# Iterating through rectangles of detected faces
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
    
#     eyes = eye.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
