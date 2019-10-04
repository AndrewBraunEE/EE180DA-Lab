import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def detect_faces(cascade, image):

    image_copy = image.copy()

    #grey scale for better calculation
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    #using cascade library
    faces_rect = cascade.detectMultiScale(gray_image, 1.3, 5)
    #scaleFactor = 1.3, minNeighbors=5

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image_copy, len(faces_rect)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    image_processed, num_faces = detect_faces(haar_cascade_face, frame)
    print('Num of Faces(0 means not a face): ' + str(num_faces))
    # Display the resulting frame
    cv2.imshow('img',image_processed)
    #press x to quit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()