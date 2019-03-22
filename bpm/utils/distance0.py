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
  print("q.shape:", array1.shape)
  print("p.shape:", array2.shape)
  np.save("q.npy", array1)
  np.save("g.npy", array2)
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
    gf_0 = gf[:, 0:256]
    gf_1 = gf[:, 256:512]
    gf_2 = gf[:, 512:768]
    gf_3 = gf[:, 768:1024]
    gf_4 = gf[:, 1024:1280]
    gf_5 = gf[:, 1280:1536]

    qf_0 = qf[:, 0:256]
    qf_1 = qf[:, 256:512]
    qf_2 = qf[:, 512:768]
    qf_3 = qf[:, 768:1024]
    qf_4 = qf[:, 1024:1280]
    qf_5 = qf[:, 1280:1536]
    batchsize_q = qf.shape[0]
    batchsize_g = gf.shape[0]
    d = np.zeros([6, 6, batchsize_q, batchsize_g])
    for m in range(0, 6):
        for n in range(0, 6):
            d[m][n]=cdist(qf[:, 256*m:256*m+256], gf[:, 256*n:256*n+256])
    d1 = np.zeros([6, batchsize_q, batchsize_g])
    for m in range(0, 6):
        for i in range(0, batchsize_q):
            for j in range(0, batchsize_g):
                d1[m][i][j] = min(d[m][0][i][j], d[m][1][i][j], d[m][2][i][j], d[m][3][i][j], d[m][4][i][j], d[m][5][i][j])
    dis = d1[0] + d1[1] + d1[2] + d1[3] + d1[4] + d1[5] 
    #dis = np.square(d1[1])+np.square(d1[2])+np.square(d1[3])+np.square(d1[4])+np.square(d1[5]) +np.square(d1[0])
    return dis
