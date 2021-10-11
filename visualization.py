import heapq
import math
from numba.np.ufunc import parallel
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine
from tqdm import tqdm
from random import randint
from numba import njit, guvectorize, cuda, float64, config, prange

from util import create_dirs, save_image, save_to_csv

cuda.select_device(1)
config.THREADING_LAYER = 'threadsafe'

def closest_n_vectors(h1, h2, i1, i2, n):
    i1 = Image.open(i1)
    i2 = Image.open(i2)

    # exclude points where radius is over border of image
    h1['xys'], h2['xys'], h1['desc'], h2['desc'] = remove_overflow_points(
        h1['xys'], h2['xys'], h1['desc'], h2['desc'], i1.size, i2.size)

    # find closest from each  h1 to closest in h2
    n_smallest = closest_vector_handler(h1, h2, 10)

    # visualize
    i1, i2 = visualize(i1, i2, n_smallest)

    # stitch images together
    stitched_image = stitch_images(i1, i2)

    # save
    return stitched_image, n_smallest
    
def closest_same_closest_diff(same_hotels, same_rooms, diff_hotels, diff_rooms):
    pass

def visualize(i1, i2, n_smallest):
    draw1 = ImageDraw.Draw(i1)
    draw2 = ImageDraw.Draw(i2)

    r = 3
    for s in n_smallest:
        point1, point2 = s[1][:2], s[2][:2]
        color = (randint(0, 255), randint(0,255), randint(0,255))
        circle1 = (point1[0]-r, point1[1]-r ,point1[0]+r, point1[1]+r)
        circle2 = (point2[0]-r, point2[1]-r, point2[0]+r, point2[1]+r)

        draw1.ellipse(circle1, fill=color)
        draw2.ellipse(circle2, fill=color)

    return i1, i2

def closest_vector_handler(h1, h2,n):
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

def remove_overflow_points(xys1, xys2, desc1, desc2, size1, size2):
    indices_to_be_del_1 = []
    indices_to_be_del_2 = []
    for idx1, point1 in enumerate(xys1):
        if point1[0]+point1[2] > size1[0] or point1[1]+point1[2] > size1[1] or point1[0]-point1[2] < 0 or point1[1]-point1[2] < 0:
            if idx1 >= len(xys1): continue
            indices_to_be_del_1.append(idx1)
    for idx2, point2 in enumerate(xys2):
        if point2[0]+point2[2] > size2[0] or point2[1]+point2[2] > size2[1] or point2[0]-point2[2] < 0 or point2[1]-point2[2] < 0:
            if idx2 >= len(xys2): continue
            indices_to_be_del_2.append(idx2)

    xys1 = np.delete(xys1, indices_to_be_del_1, 0)
    desc1 = np.delete(desc1, indices_to_be_del_1, 0)
    xys2 = np.delete(xys2, indices_to_be_del_2, 0)
    desc2 = np.delete(desc2, indices_to_be_del_2, 0)
    return xys1, xys2,desc1, desc2


def stitch_images(image1, image2):
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result
