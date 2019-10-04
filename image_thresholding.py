import cv2
from matplotlib import pyplot as plt

img = cv2.imread("Lab1\images\lab_puppy_dog_pictures.jpg", 0)
print(img)

ret,img1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
ret,img2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
ret,img3 = cv2.threshold(img, 100, 255, cv2.THRESH_MASK)
ret,img4 = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU)
ret,img5 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)

titles = ["Original", "BINARY", "BINARY_INV", "MASK", "OTSU", "TOZERO"]
images = [img, img1, img2, img3, img4, img5]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()