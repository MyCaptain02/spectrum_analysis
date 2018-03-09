import numpy as np
import cv2
import time

img_ratio = 1
image_default_dir = "./wv/test_12_18/image/"
img_format = ".bmp"


def image_index_origin_to_sample(row, col, sample_num):
    col_sampled = col // sample_num[1]
    row_sampled = row // sample_num[0]
    sample_seq = row % sample_num[0] * sample_num[1] + col % sample_num[1]
    return (sample_seq, (row_sampled, col_sampled))


def image_index_sample_to_origin(sample_info, sample_num):
    row = sample_info[1][0] * sample_num[0] + sample_info[0] // sample_num[1]
    col = sample_info[1][1] * sample_num[1] + sample_info[0] % sample_num[1]
    return (row, col)


def image_index_to_row_col(index, img_shape):
    row = index // img_shape[1]
    col = index % img_shape[1]
    return (row, col)


def image_row_col_to_index(row, col, img_shape):
    return row * img_shape[1] + col


def get_origin_image_info(image_path=image_default_dir + "420" + img_format):
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    return img.shape


def get_image_info(image_path=image_default_dir + "420" + img_format):
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    img = cv2.resize(img, (int(img.shape[1] / img_ratio), int(img.shape[0] / img_ratio)))
    return img.shape


def get_ratio_image(image_path= image_default_dir + "550" + img_format):
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    img = cv2.resize(img, (int(img.shape[1] / img_ratio), int(img.shape[0] / img_ratio)))
    return img


def read_image_to_row_vector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    # blur and remove the noise
    cv2.medianBlur(img, 5, img)
    # img = cv2.bilateralFilter(img, 5, 0, 0)
    img = cv2.resize(img, (int(img.shape[1] / img_ratio), int(img.shape[0] / img_ratio)))
    # cv2.imshow("good", img)
    img_row_vector = img.reshape(1, img.size)
    return img_row_vector


def construct_spectrum_vectors(path=image_default_dir, min_wvlen=420, max_wvlen=720, wvlen_step=10):
    """
    :return: image vector with n * k, n is the num of pixel and k is the num of spectrum
    """
    img_shape = get_image_info(path + str(min_wvlen) + img_format)
    spectrum_num = (max_wvlen - min_wvlen) // wvlen_step + 1
    img_vector = np.zeros(shape=(spectrum_num, img_shape[0] * img_shape[1]))
    for i, wv in enumerate(range(min_wvlen, max_wvlen + wvlen_step, wvlen_step)):
        row_vector = read_image_to_row_vector(path + str(wv) + img_format)
        img_vector[i] = row_vector
    return img_vector.T


from scipy.spatial.distance import pdist
def adjacency_matrix(img_vector):
    '''Compute the distance between two pixels.

    :param img_vector: image vector (n * d, n is rows and d is columns, equivalently n is the
                    number of pixels of the image, d is the spectral dimension)
    :return: row vector representation of adjacency matrix
    '''
    adj = pdist(img_vector)
    return adj


def trim_adjacency_matrix(adj, r=None, rq=.1):
    '''Make distance which is larger than the rth distance equal to zero.

    :param adj: row vector representation of adjacency matrix.
    :param r: the limit of selected distance.
    :param rq: the percentage limit of selected distance.
    :return: trimmed_adj: trimmed adjacency matrix,
             r: the limit of selected distance.
    '''
    if r is None:
        q = int(np.floor(len(adj) * rq))
        # r = np.sort(adj)[q]
        r = np.partition(adj, q)[q]   # just calculate the kth number, not to sort
    trimmed_adj = adj.copy()
    trimmed_adj[adj > r] = 0
    return trimmed_adj  #, r


from itertools import combinations
import networkx as nx
def construct_graph(adj_matrix, n):
    '''Construct a graph from the adjacency matrix.

    :param adj_matrix: adjacency matrix.
    :param n: the number of pixels of one image.
    :return: constructed graph.
    '''
    # start = time.clock()
    g = nx.Graph()
    for index, ij in enumerate(combinations(range(n), 2)):
        d = adj_matrix[index]
        if d:
            i, j = ij
            g.add_edge(i, j, weight=d)
    # end = time.clock()
    # print("construct graph time is %fs" % (end - start))
    return g


def flag_anomalies(g, min_percent_bgnd=0.1, node_colors={'anomalies':'r', 'background':'b'}):
    min_pts_bgnd = g.number_of_nodes() * min_percent_bgnd
    res = {'anomalies':[],'background':[]}
    for c in nx.connected_components(g):
        if len(c) < min_pts_bgnd:
            res['anomalies'].extend(c)
        else:
            res['background'].extend(c)
    for type, array in res.items():
        for node_id in array:
            g.node[node_id]['class'] = type
            g.node[node_id]['color'] = node_colors[type]
    return res, g


import pandas as pd
def calculate_anomaly_scores(classed, adj, n):
    scores = {}
    # start = time.clock()
    for a in classed['anomalies']:
        scores[a] = 0
        index = a - 1
        for row in range(a):
            if row != 0:
                index = index + n - 1 - row
            if row in classed['background']:
                if scores[a]:
                    scores[a] = np.min([scores[a], adj[index]])
                else:
                    scores[a] = adj[index]
            if row == a - 1:
                index = index + n - 1 - row
        for row in range(a + 1, n):
            if row in classed['background']:
                if scores[a]:
                    scores[a] = np.min([scores[a], adj[index]])
                else:
                    scores[a] = adj[index]
            index += 1
    # end = time.clock()
    # print("anomaly score time is %fs" % (end - start))

    '''
    # the time complexity is larger than the above one.
    scores1 = {}
    time1 = time.clock()
    for a in classed['anomalies']:
        scores1[a] = 0
        for z, ij in enumerate(combinations(range(n), 2)):
            i, j = ij
            if (i == a or j == a) and (
                i in classed['background'] or
                j in classed['background']):
                d = adj[z]
                if scores1[a]:
                    scores1[a] = np.min([scores1[a], d])
                else:
                    scores1[a] = d
    time2 = time.clock()
    print("the old score time is %fs" % (time2 - time1))
    '''

    # return pd.Series(scores)
    return scores

'''
# test function calculate_anomaly_scores
classed = {'anomalies':[1, 4], 'background':[0, 2, 3, 5]}
adj = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5])
n = 6
calculate_anomaly_scores(classed, adj, n)
'''

def sample_image(img_shape, spectrum_vector, sample_num):
    col_sample_num = sample_num[1]
    row_sample_num = sample_num[0]
    sample_count = col_sample_num * row_sample_num
    pixels_per_sample = spectrum_vector.shape[0] // sample_count
    sampled_image_sets = np.zeros(shape=(sample_count, pixels_per_sample, spectrum_vector.shape[1]))
    for group in range(sample_count):
        count = 0
        row_offset = group // col_sample_num
        col_offset = group - row_offset * col_sample_num
        for row in range(0, img_shape[0], row_sample_num):
            for col in range(0, img_shape[1], col_sample_num):
                sampled_image_sets[group][count] = spectrum_vector[(row + row_offset) * img_shape[1] + (col + col_offset)]
                count += 1

    '''
    # the following runtime is about twice as long as the above one.

    sampled_image_sets1 = np.zeros(shape=(sample_count, pixels_per_sample, spectrum_vector.shape[1]))
    sample_columns = img_shape[1] // sample_num[1]
    sample_rows = img_shape[0] // sample_num[0]
    time1 = time.clock()
    for group in range(sample_count):
        index = 0
        for sample_row in range(sample_rows):
            for sample_col in range(sample_columns):
                row, col = image_index_sample_to_origin((group, (sample_row, sample_col)), sample_num)
                sampled_image_sets1[group][index] = spectrum_vector[image_row_col_to_index(row, col, img_shape)]
                index += 1
    '''
    return sampled_image_sets


def process_sampled_image(sampled_image_sets, adj_limit_percent=.1, bgnd_low_limit_percent=0.1, cal_score_flag=False):
    classed_sets = {}
    scores_sets = {}
    for i in range(sampled_image_sets.shape[0]):
        time1 = time.clock()
        adj = adjacency_matrix(sampled_image_sets[i])
        time2 = time.clock()
        trim_adj = trim_adjacency_matrix(adj, rq=adj_limit_percent)
        time3 = time.clock()
        graph = construct_graph(trim_adj, sampled_image_sets[i].shape[0])
        time4 = time.clock()
        classed_sets[i], graph = flag_anomalies(graph, min_percent_bgnd=bgnd_low_limit_percent)
        time5 = time.clock()
        time6 = time5
        if cal_score_flag:
            scores_sets[i] = calculate_anomaly_scores(classed_sets[i], adj, sampled_image_sets.shape[1])
            time6 = time.clock()
        print("************%dth subset's process****************" % i)
        print("adjacency_matrix runtime is %fs" % (time2 - time1))
        print("trim_adjacency_matrix runtime is %fs" % (time3 - time2))
        print("construct_graph runtime is %fs" % (time4 - time3))
        print("flag_anomalies runtime is %fs" % (time5 - time4))
        print("calculate_anomaly_scores runtime is %fs" % (time6 - time5))
        print("**************************************************")
    return classed_sets, scores_sets


def color_table_bgr(val):
    if val > 221:
        color = (0, 0, 255)  # b,g,r
    elif val > 187:
        color = (0, 149, 255)
    elif val > 162:
        color = (0, 202, 255)
    elif val > 137:
        color = (0, 255, 255)
    elif val > 98:
        color = (142, 255, 113)
    elif val > 67:
        color = (255, 255, 0)
    elif val > 52:
        color = (255, 0, 0)
    elif val > 43:
        color = (226, 0, 0)
    elif val > 31:
        color = (197, 0, 0)
    elif val > 24:
        color = (168, 0, 0)
    elif val > 18:
        color = (139, 0, 0)
    else:
        color = (110, 0, 0)
    return color


def construct_scored_image(img_shape, classed_sets, scores_sets, sample_num):
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    sample_count = sample_num[0] * sample_num[1]
    sample_shape = (img_shape[0] // sample_num[0], img_shape[1] // sample_num[1])
    if scores_sets != {}:
        img[:][:] = (110, 0, 0)
    for i in range(sample_count):
        for subset_index in classed_sets[i]['anomalies']:
            subset_row_col = image_index_to_row_col(subset_index, sample_shape)
            row, col = image_index_sample_to_origin((i, subset_row_col), sample_num)
            if scores_sets != {}:
                img[row][col] = color_table_bgr(scores_sets[i][subset_index])
            else:
                img[row][col] = (255, 255, 255)
    return img

