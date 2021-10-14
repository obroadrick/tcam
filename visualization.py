import heapq
import math
from numba.np.ufunc import parallel
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine
from tqdm import tqdm
from random import randint
from numba import njit, guvectorize, cuda, float64, config, prange
import copy

from util import create_dirs, save_image, save_to_csv

cuda.select_device(1)
config.THREADING_LAYER = 'threadsafe'

def closest_n_vectors(h1, h2, i1, i2, n):
    """ 
    Takes dicts for descriptions, points, scores (h1 and h2) and paths to 
    images (i1 and i2) and n (how many closest vectors to consider),
    and computes and draws on a new image a visualization of the pairs of points.
    """
    i1 = Image.open(i1)
    i2 = Image.open(i2)

    # Exclude points where radius is over border of image.
    h1['xys'], h2['xys'], h1['desc'], h2['desc'] = remove_overflow_points(
        h1['xys'], h2['xys'], h1['desc'], h2['desc'], i1.size, i2.size)

    # Find closest from each  h1 to closest in h2.
    n_smallest = closest_vector_handler(h1, h2, 10)

    # Stitch images together, updating the n_smallest coordinates.
    image, n_smallest = stitch_images(i1, i2, n_smallest)
    print("Found {} nearest matching r2d2 points, excluding those along the image border".format(len(n_smallest)))

    # Draw points in pairs with random matching colors.
    image, colors = draw_points(image, n_smallest)

    # Draw lines between matching points with same matching colors.
    image = draw_lines(image, n_smallest, colors)

    # Draw circles with correct radius for each r2d2 point.
    image = draw_circles(image, n_smallest, colors)

    # Save.
    return image, n_smallest
    
def closest_same_closest_diff(same_hotels, same_rooms, diff_hotels, diff_rooms):
    pass

def draw_points(image, n_smallest):
    draw = ImageDraw.Draw(image)

    r = 5
    colors = []
    for s in n_smallest:
        point1, point2 = s[1][:2], s[2][:2]
        color = (randint(0, 255), randint(0,255), randint(0,255))
        circle1 = (point1[0]-r, point1[1]-r ,point1[0]+r, point1[1]+r)
        circle2 = (point2[0]-r, point2[1]-r, point2[0]+r, point2[1]+r)

        draw.ellipse(circle1, fill=color)
        draw.ellipse(circle2, fill=color)

        colors.append(color)

    return image, colors

def draw_lines(image, n_smallest, colors):
    draw = ImageDraw.Draw(image)

    assert len(colors) == len(n_smallest)

    for i in range(len(colors)):
        s = n_smallest[i]
        color = colors[i]
        point1, point2 = s[1][:2], s[2][:2]
        draw.line((point1[0], point1[1], point2[0], point2[1]), fill=color, width=3)

    return image

def draw_circles(image, n_smallest, colors):
    draw = ImageDraw.Draw(image)

    assert len(colors) == len(n_smallest)

    for i in range(len(colors)):
        s = n_smallest[i]
        color = colors[i]

        point1, point2 = s[1][:2], s[2][:2]
        r1, r2 = s[1][2]/2, s[2][2]/2
        circle1 = (point1[0]-r1, point1[1]-r1 ,point1[0]+r1, point1[1]+r1)
        circle2 = (point2[0]-r2, point2[1]-r2, point2[0]+r2, point2[1]+r2)

        draw.ellipse(circle1, fill=None)
        draw.ellipse(circle2, fill=None)

    return image

def closest_vector_handler(h1, h2,n):
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
    for idx2, point2 in enumerate(xys2):
        radius = point2[2] / 2
        if point2[0]+radius > size2[0] or point2[1]+radius > size2[1] or point2[0]-radius < 0 or point2[1]-radius < 0:
            if idx2 >= len(xys2): continue
            indices_to_be_del_2.append(idx2)

    xys1 = np.delete(xys1, indices_to_be_del_1, 0)
    desc1 = np.delete(desc1, indices_to_be_del_1, 0)
    xys2 = np.delete(xys2, indices_to_be_del_2, 0)
    desc2 = np.delete(desc2, indices_to_be_del_2, 0)
    return xys1, xys2,desc1, desc2

def stitch_images(image1, image2, nsmallest):
    """
    Resizes the image of smaller height to match the heights of the images.
    Stitches the two images together, updating all the coordinates in h1 and h2.
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size
    nsmallest_stiched = copy.deepcopy(nsmallest)

    # Resize the image with lesser height to match the other image (for viewing pleasure)
    if height1 < height2:
        # We need to scale up image1.
        new_h = height2
        proportion = new_h / height1
        new_w = int(width1 * proportion)
        image1 = image1.resize((new_w,new_h))
        width1 = new_w
        height1 = new_h
        # Update coordinates for image1. 
        # (first index 1 is to get the first im, second index 1 is to get y coord)
        for item in nsmallest_stiched:
            item[1][1] *= proportion
            item[1][0] *= proportion
    elif height2 < height1:
        # We need to scale up image2.
        new_h = height1
        proportion = new_h / height2
        new_w = int(width2 * proportion)
        image2 = image2.resize((new_w,new_h))
        width2 = new_w
        height2 = new_h
        # Update coordinates for points in image 2.
        for item in nsmallest_stiched:
            item[2][1] *= proportion
            item[2][0] *= proportion

    # Update x coordinates for points in image2.
    for item in nsmallest_stiched:
        item[2][0] += width1

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result, nsmallest_stiched
