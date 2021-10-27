"""
Functions for computations about closeness of r2d2 points in images.
"""

from numba import njit, guvectorize, cuda, float64, config, prange
import numpy as np
import math
import heapq
from tqdm import tqdm

def closest_vector_handler(h1, h2, n=.25):
    """
    Computes and returns the n closest (by cosine distance) r2d2 points between h1 and h2.
    """
    # heap? DS that may make it faster than sorting at end
    heap = []
    heapq.heapify(heap)

    M1 = h1['desc']; M2 = h2['desc']
    M1 = M1.astype(np.float64)
    M2 = M2.astype(np.float64)
    for idx1, v1 in tqdm(enumerate(M1), total=len(M1), desc='Finding Closest'):
        # array to hold answer. no return possible for this gpu shit
        ans = np.ones(128, dtype=np.float64)
        closest(v1, M2, ans)
        idx, score = int(ans[0]), ans[1]

        # same as above. This is validating that closest(x) = y and closest(y) = x
        ans = np.zeros(128, dtype=np.float64)
        closest(M2[idx], M1, ans)
        pair_idx, _ = int(ans[0]), ans[1]
        # pair_idx, _= closest(M2[idx], M1)

        if idx1 == pair_idx:
            heap.append( (score, h1['xys'][idx1], h2['xys'][idx]) )

    if n > 0 and n <= 1:
        return heapq.nsmallest(int(n* len(heap)), heap, key=lambda x: x[0])
    elif n >= 1:
        return heapq.nsmallest(int(n), heap, key=lambda x: x[0])
    # Defaults to .25 proportion
    return heapq.nsmallest(int(0.25* len(heap)), heap, key=lambda x: x[0])

@njit
def norm(v):
    s=0
    for i in v:
        s+=i**2
    return  math.sqrt(s)

@njit
def dot(v1, v2):
    p = 0
    for i in range(len(v1)):
        p += v1[i] * v2[i]
    return p

@guvectorize([(float64[:], float64[:, :], float64[:])], '(m),(n,m)->(m)', nopython=True)
def closest(v1, M2, ans):
    closest_score = 1
    closest_index = 0
    for idx in prange(len(M2)):
        v2 = M2[idx]
        score = 1- (dot(v1,v2)/ (norm(v1) * norm(v2)))
        if score < closest_score:
            closest_score = score
            closest_index = idx
    # return closest_index, closest_score
    ans[0] = closest_index
    ans[1] = closest_score

# def remove_overflow_points(xys1, xys2, desc1, desc2, size1, size2):
def remove_overflow_points(xys1, desc1, size1, xys2=None, desc2=None, size2=None):
    """
    Removes 'overflow' points, those whose radius exceeds the boundaries of the image.

    Useful because these points could artificially produce good matches since boundaries are similar.
    """
    indices_to_be_del_1 = []
    indices_to_be_del_2 = []
    for idx1, point1 in enumerate(xys1):
        radius = point1[2] / 2
        if point1[0]+radius > size1[0] or point1[1]+radius > size1[1] or point1[0]-radius < 0 or point1[1]-radius < 0:
            if idx1 >= len(xys1): continue
            indices_to_be_del_1.append(idx1)
    if xys2 is not None:
        for idx2, point2 in enumerate(xys2):
            radius = point2[2] / 2
            if point2[0]+radius > size2[0] or point2[1]+radius > size2[1] or point2[0]-radius < 0 or point2[1]-radius < 0:
                if idx2 >= len(xys2): continue
                indices_to_be_del_2.append(idx2)
        xys2 = np.delete(xys2, indices_to_be_del_2, 0)
        desc2 = np.delete(desc2, indices_to_be_del_2, 0)

    xys1 = np.delete(xys1, indices_to_be_del_1, 0)
    desc1 = np.delete(desc1, indices_to_be_del_1, 0)
    
    return xys1, xys2,desc1, desc2
