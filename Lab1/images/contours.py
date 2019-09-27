import cv2
import numpy as np
from matplotlib import pyplot as plt


def contour(image = 'something', percent_arclength = 0.1):
	img = cv2.imread(image, 0)
	original_image = cv2.imread(image, 0)
	ret, thresh = cv2.threshold(img, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	M = cv2.moments(cnt)
	#M is the number of moment values calculated

	#You can then calculate centroids from the following lines:
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])

	#Contour areas, perimeter, and approximation
	area = cv2.contourArea(cnt)
	perimeter = cv2.arcLength(cnt, True)
	
	#10% of arclength
	epsilon = percent_arclength*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)

	#ConvexHull
	hull = cv2.convexHull(cnt)
	#Convexity, k
	k = cv2.isContourConvex(cnt)

	print_dict = dict(zip(["M", "cx", "cy", "area", "perimeter", "epsilon", "k"] , [M, cx, cy, area, perimeter, epsilon, k]))
	print(str(print_dict))

	#Bounding Rectangle
	x,y,w,h = cv2.boundingRect(cnt)
	img = cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imwrite('output/BoundingRect.jpg', img)

	#Rotated Bounding Rectangle
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	im = cv2.drawContours(original_image,[box],0,(0,0,255),2)

	cv2.imwrite('output/BoundingRectRotated.jpg', im)

	#Minimum Enclosing Circle
	(x,y),radius = cv2.minEnclosingCircle(cnt)
	center = (int(x),int(y))
	radius = int(radius)
	img = cv2.circle(img,center,radius,(0,255,0),2)
	cv2.imwrite('output/MinCircle.jpg', img)

	#Fitting Ellipse
	ellipse = cv2.fitEllipse(cnt)
	im = cv2.ellipse(original_image,ellipse,(0,255,0),2)
	cv2.imwrite('output/FitEllipse.jpg', im)
	
	#Fitting a line
	rows,cols = img.shape[:2]
	[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
	lefty = int((-x*vy/vx) + y)
	righty = int(((cols-x)*vy/vx)+y)
	img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
	cv2.imwrite('output/FitEllipse.jpg', img)


def main():
	contour(image = 'star-c.png')


if __name__ == '__main__':
	main()