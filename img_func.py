import cv2
from matplotlib import pyplot as plt
import numpy as np

def show_hist(img):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_ANYDEPTH)

    plt.subplot()
    plt.hist(img.ravel(), 256, [0, 4096])
    plt.title("img_hist")
    plt.show()

def img_16_to_8_bits(src, type="exp"):
    rows, cols = src.shape
    dst = np.zeros((rows, cols))

    if type == "exp":
        for row in range(rows):
            for col in range(cols):
                dst[row, col] = src[row, col] ** (2 / 3)
    elif type == "line":
        x1 = 512
        y1 = 128
        x2 = 3072
        y2 = 224
        for row in range(rows):
            for col in range(cols):
                if src[row, col] < x1 :
                    dst[row, col] = y1 / x1 * src[row, col]
                elif src[row, col] < x2 :
                    dst[row, col] = (y2 - y1) / (x2 - x1) * (src[row, col] - x1) + y1
                else:
                    dst[row, col] = (255 - y2) / (4095 - x2) * (src[row, col] - x2) + y2

    return dst

wavelen = 550
file = "./wv/" + str(wavelen) + '.png'
file_show = "./wv/" + str(wavelen) + '.bmp'
show_hist(file)

img_origin = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
img_8bits = img_16_to_8_bits(img_origin, type='line')
img_show = cv2.imread(file_show, cv2.IMREAD_GRAYSCALE)

plt.subplot(121)
plt.title("process to 8bit")
plt.imshow(img_8bits, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.title("high 8bits of raw")
plt.imshow(img_show, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.show()

