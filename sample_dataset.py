""" Sample and label """
from __future__ import unicode_literals
import os
import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from util import open_pckl_5_file, write_pckl_file
import faiss


def sample():
    random.seed(42)

    R2D2 = open_pckl_5_file('/pless_nfs/home/mdt_/tcam/datasets/0.1k/r2d2_objs/clean.pckl')

    hotel_ids = R2D2.get_hotel_ids()

    # Sample 10 random hotels uniformly
    hotel_ids = random.sample(hotel_ids, 10)

    # Sample *all* images (let's not miss the painting or nice angle of lamp in one of them)
    proportion_of_points_to_sample = .25

    sample = {}
    sample['hotel_pt_xys'] = []
    sample['hotel_pt_desc'] = []
    sample['hotel_pt_iids'] = []
    sample['hotel_pt_hids'] = []
    # sample['path'] = []

    for hid in tqdm(hotel_ids, desc='Sampling Dataset'):
        image_ids = R2D2.get_room_ids(hid) # these are actually image id's :) research code!
        feature_file = R2D2.open_and_extract_feature_file(hid)

        for im_id in image_ids:
            # Sample uniformly .25 of all points per image (which gives a sample of .25 of the points per hotel)
            # points = list(zip(feature_file[im_id]['desc'], feature_file[im_id]['xys'][:, :2]))
            
            
            random.seed(42)
            idxs = [i for i in range(len(feature_file[im_id]['desc']))]
            n = len(feature_file[im_id]['desc'])
            sample_num = int(n*proportion_of_points_to_sample)
            points_idxs = random.sample(idxs, sample_num)

            for i in points_idxs:
                sample['hotel_pt_desc'].append(feature_file[im_id]['desc'][i])
                sample['hotel_pt_iids'].append(int(im_id))
                sample['hotel_pt_xys'].append(feature_file[im_id]['xys'][i])
                sample['hotel_pt_hids'].append(hid)
                # sample['path'].append(R2D2.get_path_to_image(0, im_id))
    return sample
    
def label(s):

    iids = np.array(s['hotel_pt_iids'])
    points = np.array(s['hotel_pt_desc'])
    hids = np.array(s['hotel_pt_hids'])
    xys = np.array(s['hotel_pt_xys'])
    del s

    distaces_same = np.zeros_like(hids)
    distaces_diff = np.zeros_like(hids)
    closest_same_image_id = np.zeros_like(hids)
    closest_diff_image_id = np.zeros_like(hids)
    xys_same = np.zeros_like(hids)
    xys_diff = np.zeros_like(hids)  
    diff_hid = np.zeros_like(hids)

    unique_hids, unique_iids = list(np.unique(hids)), list(np.unique(iids))
    for hid in tqdm(unique_hids):
        diff_points = points[np.argwhere(hids != hid).reshape((-1,))]
        diff_index = faiss.IndexFlatIP(128)
        train = np.ascontiguousarray(diff_points,dtype=np.float32)
        faiss.normalize_L2(train)
        diff_index.add(train)

        for iid in unique_iids:
            same_hotel_idx = np.argwhere((hids == hid) & (iids != iid)).reshape((-1,))
            query_point_idx = np.argwhere((iids == iid)).reshape((-1,))

            query_points = points[query_point_idx]
            same_points = points[same_hotel_idx]


            same_index = faiss.IndexFlatIP(128)
            train = np.ascontiguousarray(same_points,dtype=np.float32)
            faiss.normalize_L2(train)
            same_index.add(train)

            # for qp in same_points:
            same_d, same_i = same_index.search(query_points, 1)

            diff_d, diff_i = diff_index.search(query_points, 1)
            
            same_d, diff_d = same_d.reshape((-2,)),diff_d.reshape((-2,))

            # ratio = diff_d / same_d
            # ratio = ratio.reshape((-2,))
            # ratios[same_hotel_idx] = ratio

            # ratio[ratio >= cutoff] = 1
            # ratio[ratio < cutoff] = 0

            # labels[same_hotel_idx] = ratio

            distaces_same[query_point_idx] = same_d
            distaces_diff[query_point_idx] = diff_d

            closest_same_image_id[query_point_idx] = iids[same_i]
            closest_diff_image_id[query_point_idx] = iids[diff_i]

            xys_same[query_point_idx] = xys[same_i]
            xys_diff[query_point_idx] = xys[diff_i]

            diff_hid[query_point_idx] = hids[diff_i]




    dataset = pd.DataFrame(points)
    # dataset['label'] = labels
    # dataset['ratios'] = ratios
    dataset['distance_same'] = distaces_same
    dataset['distance_diff'] = distaces_diff
    dataset['hid'] = hids
    dataset['iid'] = iids
    dataset['xys'] = np.array(s['hotel_pt_xys'])
    dataset['xys_same'] = xys_same
    dataset['xys_diff'] = xys_diff
    dataset['closest_diff_hid'] = diff_hid
    dataset['closest_same_iid'] = closest_same_image_id
    dataset['cloesest_diff_iid'] = closest_diff_image_id
    dataset.to_pickle('/pless_nfs/home/mdt_/tcam/better_datasets_cause_we_are_dumb/better_dataset.pckl')
    dataset.to_csv('/pless_nfs/home/mdt_/tcam/better_datasets_cause_we_are_dumb/better_dataset.csv')



    

def main():
    
    s = sample()
    label(s)

if __name__ == '__main__':
    main()









