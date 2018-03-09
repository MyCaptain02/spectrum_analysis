import cv2
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from tad_img import *

sample_num = (32, 32)
adj_limit_percent = 0.05    # 像素点间距离由小到大排序，两像素点距离在该值代表的距离以下的才作为边加入图中
bgnd_low_limit_percent = 0.6    # 连通图中的像素点数所占总数百分比不小于该值为背景像素
score_flag = True

start = time.clock()

img_vector = construct_spectrum_vectors()
img_shape = get_image_info()
sampled_img_sets = sample_image(img_shape, img_vector, sample_num)
classed_sets, scores_sets = process_sampled_image(sampled_img_sets, adj_limit_percent, bgnd_low_limit_percent, score_flag)

time1 = time.clock()
img = construct_scored_image(img_shape, classed_sets, scores_sets, sample_num)
time2 = time.clock()
print("the construct image time is %fs" % (time2 - time1))
end = time.clock()
print("the total runtime is %fs" % (end - start))

img_origin = get_ratio_image()

cv2.imshow("processed image", img)
# img_show = np.concatenate((img_origin, img))
# cv2.imshow("good", img_show)
cv2.waitKey()
cv2.destroyAllWindows()

print("end for breakpoints")