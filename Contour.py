import cv2
import numpy as np


img = cv2.imread('Nike2.jpg',0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

#print(contours[0])

cnt = contours[0]
M = cv2.moments(cnt)
print(M)
#1

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#2

area = cv2.contourArea(cnt)

#3

perimeter = cv2.arcLength(cnt,True)

#4

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

#5


hull = cv2.convexHull(cnt)

#6

k = cv2.isContourConvex(cnt)

#7a

x,y,w,h = cv2.boundingRect(cnt)
img7a = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

cv2.imwrite('output/BoundingRect.jpg', img7a)

#7b

img = cv2.imread('Nike2.jpg',0)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img7b = cv2.drawContours(img,[box],0,(0,0,255),1)

cv2.imwrite('output/RotationRect.jpg', img7b)

#8

img = cv2.imread('Nike2.jpg',0)

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img8 = cv2.circle(img,center,radius,(0,255,0),1)

cv2.imwrite('output/MinCircle.jpg', img8)

#9

img = cv2.imread('star.jpg',0)

ellipse = cv2.fitEllipse(cnt)
img9 = cv2.ellipse(img,ellipse,(0,255,0),2)

cv2.imwrite('output/FitEllipse.jpg', img9)

#10
img = cv2.imread('Nike2.jpg',0)
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img10 = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
cv2.imwrite('output/FitLine.jpg', img10)




