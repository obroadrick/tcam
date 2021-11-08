import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from closeness_computations import closest_vector_handler
from util import open_pckl_file, create_dirs

from tqdm import tqdm
from random import sample, seed

seed(42)

import warnings
warnings.filterwarnings("ignore")
"""
Functions for our initial, naive search that we will use to build an 
understanding of how r2d2 points can be used for our matching problem.
"""

from closeness_computations import closest_vector_handler


def compute_matches_for_image_across_hotels(r2d2_features, hid, rid):
    """
    This function is the initial "search" algorithm that we are implementing.
    For the query image, we compute the pairwise-closest r2d2 points for a sample 
    of a images from the same hotel and a sample of images from a different hotel.
    We then save and plot the distribution of cosine-similarity for these two classes.

    hid     hotel id of the query image
    rid     room id of the query image
    """
    # Get the query image features
    query_img_features = r2d2_features.open_and_extract_feature_file(hid, rid)

    # SAME HOTEL
    # Get a sample of images from the same hotel
    same_hotel_rids = r2d2_features.get_room_ids(hid)
    same_hotel_feature_lists = []
    for same_rid in same_hotel_rids:
        if same_rid == rid: continue
        same_hotel_feature_lists.append(r2d2_features.open_and_extract_feature_file(hid, same_rid))
    
    # Compute the cosine similarities between the query image and sampled images from the same hotel
    same_hotel_matches = []
    for same_hotel_features in same_hotel_feature_lists:
    # print(closest_vector_handler(query_img_features, same_hotel_features, 1)[0])
        same_hotel_matches.extend([1-i[0] for i in closest_vector_handler(query_img_features, same_hotel_features, 1)])

    # DIFFERENT HOTEL
    # Get a sample of images from a different hotel
    # Initially, let's just do this for a single different hotel... could wrap in loop for a sample of the total hotels too though
    diff_hotel_hids = []
    for id in r2d2_features.get_hotel_ids():
        if id != hid:
            diff_hotel_hids.append(id)
    diff_hotel_feature_lists = []
    for idx, diff_hid in enumerate(list(sample(diff_hotel_hids, 100))):
        ff = r2d2_features.open_and_extract_feature_file(diff_hid)
        diff_rid = int(sample(ff.keys(), 1)[0])
        diff_hotel_feature_lists.append(ff[diff_rid])

    # Compute the cosine similarities between the query image and sampled images from the diff hotel
    diff_hotel_matches = []
    for diff_hotel_features in diff_hotel_feature_lists:
        diff_hotel_matches.extend([1-i[0] for i in closest_vector_handler(query_img_features, diff_hotel_features, 1)])

    # Save this data
    dir = 'datasets/0.1k/naive_search_results_more_bins/{}/'.format(hid)
    create_dirs(dir)
    array = np.array([same_hotel_matches, diff_hotel_matches],dtype=object)
    np.save(dir+str(hid)+'_'+str(rid)+'.npy', array)
    plot(same_hotel_matches, diff_hotel_matches, hid, rid, dir)
    return same_hotel_matches, diff_hotel_matches

def plot(same_hotel_matches, diff_hotel_matches, hid, rid, dir):
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1])
    for a in [(same_hotel_matches, 'same'), (diff_hotel_matches, 'diff')]:
        sns.distplot(a[0],label=a[1], bins=100, ax=ax, kde=True, norm_hist=True)
    plt.title('HID={}; RID={}'.format(hid, rid))
    plt.legend()
    plt.savefig(dir+str(hid)+'_'+str(rid)+'.png')
    plt.close()

"""
# NOTES
def closest_vector_handler(h1, h2,n):
Computes and returns the n closest (by cosine distance) r2d2 points between h1 and h2.
"""
def main(path_to_r2d2):
    #cleaned_data = open_pckl_file(path_to_sampled)
    r2d2 = open_pckl_file(path_to_r2d2)
    
    hids = r2d2.get_hotel_ids()

    for hid in tqdm(sample(hids, 100)):
        same, diff = [], []
        for rid in r2d2.get_room_ids(hid):
           s, d = compute_matches_for_image_across_hotels(r2d2, hid, rid)
        same.extend(s), diff.extend(d)
        dir = 'datasets/0.1k/naive_search_results_more_bins/{}/'.format(hid)
        plot(same, diff, hid, 'OVERALL', dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  parser.add_argument('-d', type=str, required=True) # path to sampled
    parser.add_argument('-p', type=str, required=True) # path to r2d2 obj
    args = parser.parse_args()

    main(args.p)
