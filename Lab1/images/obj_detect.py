import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def detect_faces(cascade, image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy, len(faces_rect)

def main():
	#Finds the number of faces in the current webcam-generated image frame, and paints a green rectangle around the face.
	video_capture = None
	try:
		video_capture = cv2.VideoCapture(0)
		while True:
			if not video_capture.isOpened():
				raise Exception("Webcam was not found!")
			ret, frame = video_capture.read()
			#frameRGB = frame[:,:,::-1] #RGB instead of BGR
			haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
			image_processed, num_faces = detect_faces(haar_cascade_face, frame)
			print('Faces found: ' + str(num_faces))
			cv2.imshow('img', image_processed)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	except KeyboardInterrupt:
		video_capture.release()
		pass

if __name__ == '__main__':
	main()