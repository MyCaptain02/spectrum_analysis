import numpy as np
from itertools import combinations
import networkx as nx

def trim_adjacency_matrix(adj, r=None, rq=.1):
    if r is None:
        q = int(np.floor(len(adj) * rq))
        print("q: %d" % q)
        r = np.sort(adj)[q]
    print("r: %d" % r)
    adj2 = adj.copy()
    adj2[adj > r] = 0
    return adj2, r

def construct_graph(edges, n):
    g = nx.Graph()
    for z, ij in enumerate(combinations(range(n), 2)):
        d = edges[z]
        if d:
            i, j = ij
            g.add_edge(i, j, weight=d)
    return g

def flag_anomalies(g, min_pts_bgnd, node_colors={'anomalies':'r', 'background':'b'}):
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
    for a in classed['anomalies']:
        scores[a] = 0
        for z, ij in enumerate(combinations(range(n),2)):
            i,j = ij
            if (i == a or j == a) and (
                i in classed['background'] or
                j in classed['background']):
                d = adj[z]
                if scores[a]:
                    scores[a] = np.min([scores[a], d])
                else:
                    scores[a] = d
    return pd.Series(scores)

from scipy.spatial.distance import pdist
def tad_classify(X, method='euclidean', r=None, rq=.1, p=.1, distances=None):
    if not distances:
        adj = pdist(X, method)
    edges, r = trim_adjacency_matrix(adj, r, rq)
    n = X.shape[0]
    g = construct_graph(edges, n)
    classed, g =  flag_anomalies(g, n*p)
    scores = calculate_anomaly_scores(classed, adj, n)
    return {'classed':classed, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*p, 'distances':adj}


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
res = tad_classify(df)
    
df['anomaly'] = 0
df.anomaly.ix[res['classed']['anomalies']] = 1
scatter_matrix(df.ix[:, :4], c=df.anomaly, s=(25 + 50 * df.anomaly), alpha=.8)
plt.show()

from sklearn.decomposition import PCA
g = res['g']
X_pca = PCA().fit_transform(df)
pos = dict((i,(X_pca[i,0], X_pca[i,1])) for i in range(X_pca.shape[0]))
colors = [node[1]['color'] for node in g.nodes(data=True)]
labels = {}
for node in g.nodes():
    if node in res['classed']['anomalies']:
        labels[node] = node
    else:
        labels[node] = ''
nx.draw(g, pos=pos, node_color = colors, labels=labels)
plt.show()