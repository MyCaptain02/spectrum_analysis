import cv2
from matplotlib import pyplot as plt
import numpy as np

img_origin = cv2.imread("./wv/550.png", cv2.IMREAD_ANYDEPTH)
#img_show1 = cv2.addWeighted(img_origin, 0.0625, img_origin, 0, 0)
img_origin = cv2.medianBlur(img_origin, 5)
img_high_8bits = img_origin >> 4
img_high_8bits = img_high_8bits.astype(np.uint8)
img_show = img_high_8bits.astype(np.uint8)

# hist_img_show = cv2.calcHist(img_show, [0], None, [256], [0, 256])
cv2.equalizeHist(img_show, img_show)

plt.subplot(211)
plt.hist(img_high_8bits.ravel(), 256, [0, 256])
plt.title("img_high_8bits")
plt.subplot(212)
plt.hist(img_origin.ravel(), 256, [0, 4096])
plt.title("img_show")
plt.show()


# kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))
# img_show = cv2.morphologyEx(img_show, cv2.MORPH_OPEN, kernel)
plt.subplot(121)
plt.imshow(img_show, cmap='gray')
plt.title("img_show")
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(img_high_8bits, cmap='gray')
plt.title("img_high_8bits")
plt.xticks([]), plt.yticks([])
plt.show()