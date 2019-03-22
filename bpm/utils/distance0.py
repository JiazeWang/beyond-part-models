"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np
from scipy.spatial.distance import cdist


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    dist = reranknew(array1, array2)
    # shape [m1, 1]]
    """
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    """
    return dist


def reranknew(qf, gf):
    print("Reranking new now!!!!!!")
    qf_0 = qf[:, 0:2048]
    qf_1 = qf[:, 2048:4096]
    qf_2 = qf[:, 4096:6144]
    qf_3 = qf[:, 6144:8192]
    qf_4 = qf[:, 8192:10240]
    qf_5 = qf[:, 10240:12288]

    gf_0 = gf[:, 0:2048]
    gf_1 = gf[:, 2048:4096]
    gf_2 = gf[:, 4096:6144]
    gf_3 = gf[:, 6144:8192]
    gf_4 = gf[:, 8192:10240]
    gf_5 = gf[:, 10240:12288]

    batchsize_q = qf.shape[0]
    batchsize_g = gf.shape[0]
    d = np.zeros([6, 6, batchsize_q, batchsize_g])
    for m in range(0, 6):
        for n in range(0, 6):
            d[m][n]=cdist(qf[:, 2048*m:2048*m+2048], gf[:, 2048*n:2048*n+2048])
    d1 = np.zeros([6, batchsize_q, batchsize_g])
    for m in range(0, 6):
        for i in range(0, batchsize_q):
            for j in range(0, batchsize_g):
                d1[m][i][j] = min(d[m][0][i][j], d[m][1][i][j], d[m][2][i][j], d[m][3][i][j], d[m][4][i][j], d[m][5][i][j])
        #dis = d1 + d2 + d3 + d4 + d5 + d6 + d7
    dis = np.square(d1[1])+np.square(d1[2])+np.square(d1[3])+np.square(d1[4])+np.square(d1[5]) +np.square(d1[0])
    return dis
