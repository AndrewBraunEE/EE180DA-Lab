import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr = cv2.imread("Lab1\images\lab_puppy_dog_pictures.jpg", 1)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gray_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

img1 = np.concatenate((bgr, gray_3_channel), axis=1)
img2 = np.concatenate((img1, hsv), axis=1)

cv2.imshow("yes", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Lab1\images\Changin_Colorspace.jpg", img2)