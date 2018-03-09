import numpy as np

def rx_detector(img_vector):
    """
    :param img_vector: k * n matrix, k is the spectrum num while n is pixel num
    :return:
    """
    pixel_mean = np.mean(img_vector, axis=1)
    # pixel_minus_mean = img_vector - pixel_mean
    cov_matrix = np.cov(img_vector)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    rx_img = np.zeros(img_vector.shape[1], dtype=np.uint8)
    columns = img_vector.shape[1]
    for col in range(columns):
        vec = img_vector[:, col] - pixel_mean
        rx_img[col] = vec.T.dot(cov_matrix_inv.dot(vec))
    return rx_img

def construct_rx_img(rx_img, percent = 0.25):
    q = int(np.floor(percent * len(rx_img)))
    r = np.partition(rx_img, q)[q]
    rx_img_copy = rx_img.copy()
    rx_img_copy[rx_img >= r] = 255
    # rx_img_copy[rx_img < r] = 0
    return rx_img_copy

import tad_img
img = tad_img.construct_spectrum_vectors("./wv/test_12_18/image/").T
img_shape = tad_img.get_origin_image_info()
rx_img = rx_detector(img)
result = construct_rx_img(rx_img)
import cv2

# save images with different threshold percent for setting value of anomaly pixel.
# for i in range(1, 100, 1):
#     p = i / 100
#     temp = construct_rx_img(rx_img, p)
#     temp = temp.reshape(img_shape)
#     cv2.imwrite("./" + str(i) + "%.png", temp)

img_show = result.reshape(img_shape)
cv2.imshow("rx_img", img_show)
cv2.waitKey()
cv2.destroyAllWindows()