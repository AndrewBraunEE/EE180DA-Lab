import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_edge(img_file = 'lab_puppy_dog_pictures'):
	img = cv2.imread('lab_puppy_dog_pictures.jpg')
	edges = cv2.Canny(img, 100, 200)


	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()



def main():
	detect_edge()


if __name__ == '__main__':
	try:
		main()
