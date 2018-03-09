import cv2
import numpy as np
from matplotlib import pyplot as plt

blue = cv2.imread("wv/430.bmp", cv2.IMREAD_GRAYSCALE)   # 435.8nm
green = cv2.imread("wv/550.bmp", cv2.IMREAD_GRAYSCALE)   # 546.1nm
red = cv2.imread("wv/700.bmp", cv2.IMREAD_GRAYSCALE)    # 700nm

mv = [blue, green, red]
dst = cv2.merge(mv)

cv2.imshow("test", dst)
cv2.waitKey(0)