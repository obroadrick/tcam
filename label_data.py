""" Labels the sampled data for training the binary classifier of *good* and *bad* points."""
import numpy as np
from numpy.lib.function_base import diff
# from scipy.spatial.distance import cosine
# import scipy.spatial as sp
from scipy.spatial.distance import cdist
from util import open_pckl_file, create_dirs
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset = open_pckl_file('datasets/0.1k/datasets/10_hotel_sample.pckl')


# the closest points in the same hotel and in a different hotel, in order of points appearance in the query image
for hid in tqdm(list(dataset.keys())):
    hotel = []
    for qid in dataset[hid].keys():
        # qid is the image id of our query image
        # now we compare points in qid to same hotel points
        c_same, c_diff = [], []
        same_points = []
        same_point_xys = []
        same_point_iids = []
        for iid in dataset[hid].keys():
            if iid == qid: continue
            for point in dataset[hid][iid]:
                same_points.append(point[0])
                same_point_xys.append(point[1])
                same_point_iids.append(iid)
        diff_points = []
        diff_point_xys = []
        diff_point_hids = []
        diff_point_iids = []
        for diff_hid in dataset.keys():
            if hid == diff_hid: continue
            for diff_iid in dataset[diff_hid].keys():
                diff_points = []
                for diff_point in dataset[diff_hid][diff_iid]:
                    diff_points.append(diff_point[0])
                    diff_point_xys.append(diff_point[1])
                    diff_point_hids.append(hid)
                    diff_point_iids.append(diff_iid)
        for query_point in dataset[hid][qid]:
            # get descriptor of this point
            query_point = query_point[0]
            same_points = np.array(same_points)
            query_point = np.array(query_point)
            query_point = np.reshape(query_point, (1,128))
            same_dists = cdist(same_points, query_point, 'cosine')
            diff_dists = cdist(diff_points, query_point, 'cosine')

            # find index for closest points in each class (cluster, category, whatever it is called)
            s_idx = np.argmin(same_dists)
            d_idx = np.argmin(diff_dists)

            # track for each query point just the data for the closest point
            # we track: (distance, descriptor, xys, hid, iid)
            c_same.append(np.array([float(same_dists[s_idx]), \
                            same_points[s_idx], \
                            same_point_xys[s_idx], \
                            qid, \
                            same_point_iids[s_idx]], dtype='object'))
            c_diff.append(np.array([float(diff_dists[d_idx]), \
                            diff_points[d_idx], \
                            diff_point_xys[d_idx], \
                            diff_point_hids[d_idx], \
                            diff_point_iids[d_idx]], dtype='object'))
            
    hotel.append(np.array([c_same, c_diff], dtype='object'))
            # print(c_same[-1][0], c_diff[-1][0], c_same[-1][0] / c_diff[-1][0])

        
        # now we compare points in iid to different hotel poitns
    save_dir = 'datasets/0.1k/datasets/same_diff_hotel/'
    np.save(save_dir+'/{}'.format(hid), hotel)
    #print(np.mean(np.array(c_same)), np.mean(np.array(c_diff)))    
        # we store the nearest point in same hotel, and nearest point in diff hotel, and those two distances
        # we want to be able to reaccess those specific points so we need to know their location


