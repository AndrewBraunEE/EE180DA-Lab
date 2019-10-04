import cv2
from matplotlib import pyplot as plt

img = cv2.imread("Lab1\images\pic1.png", 0)
edges = cv2.Canny(img, 100, 200)

images = [img, edges]
titles = ["Original", "Edge Detection"]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
