""" Labels the sampled data for training the binary classifier of *good* and *bad* points."""
import numpy as np
from numpy.lib.function_base import diff
# from scipy.spatial.distance import cosine
# import scipy.spatial as sp
from scipy.spatial.distance import cdist
from util import open_pckl_file, write_pckl_file
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset = open_pckl_file('datasets/0.1k/datasets/10_hotel_sample.pckl')

# the closest points in the same hotel and in a different hotel, in order of points appearance in the query image
for hid in tqdm(list(dataset.keys())):
    # the goal here, for this particular hotel, is to populate for each query points (q_pts) the nearest
    # point in the same hotel (c_same) and the nearest point in a diff hotel (c_diff)
    q_pts, c_same, c_diff = [], [], []

    diff_points = []
    diff_point_xys = []
    diff_point_hids = []
    diff_point_iids = []
    for diff_hid in dataset.keys():
        if hid == diff_hid:
            continue
        for diff_iid in dataset[diff_hid].keys():
            for diff_point in dataset[diff_hid][diff_iid]:
                diff_points.append(diff_point[0])
                diff_point_xys.append(diff_point[1])
                diff_point_hids.append(hid)
                diff_point_iids.append(diff_iid)
    # array of all the points from different hotels
    diff_points = np.array(diff_points)
    for qid in list(dataset[hid].keys()):  # for image id
        # qid is the image id of our query image
        # now we compare points in qid to same hotel points
        same_points = []
        same_point_xys = []
        same_point_iids = []
        for iid in dataset[hid].keys():
            if iid == qid:
                continue
            for point in dataset[hid][iid]:
                same_points.append(point[0])
                same_point_xys.append(point[1])
                same_point_iids.append(iid)
        # this is an array of many points (all from the same hotel)
        same_points = np.array(same_points)
        for query_point_item in dataset[hid][qid]:
            # get a 2d array with just the descriptor of the one query point
            query_point = np.reshape(np.array(query_point_item[0]), (1, 128))

            # compute the cosine distances from the query point to the same and diff points
            same_dists = cdist(same_points, query_point, 'cosine')
            diff_dists = cdist(diff_points, query_point, 'cosine')

            # find index for nearest points in each class
            s_idx = np.argmin(same_dists)
            d_idx = np.argmin(diff_dists)

            # track for each query point just the data for the closest points in same and diff classes
            # we track: (distance, descriptor, xys, hid, iid)
            q_pts.append(query_point_item[0])
            c_same.append(np.array([float(same_dists[s_idx]), same_points[s_idx],
                          same_point_xys[s_idx], hid, same_point_iids[s_idx]], dtype='object'))
            c_diff.append(np.array([float(diff_dists[d_idx]), diff_points[d_idx], diff_point_xys[d_idx],
                          diff_point_hids[d_idx], diff_point_iids[d_idx]], dtype='object'))

    # save this label data for this hotel
    save_dir = 'datasets/0.1k/datasets/same_diff_hotel/{}.pckl'.format(hid)
    write_pckl_file(save_dir, [np.array(q_pts), np.array(c_same), np.array(c_diff)])
