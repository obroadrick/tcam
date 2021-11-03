import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from closeness_computations import closest_vector_handler
from util import open_pckl_file

from tqdm import tqdm
from random import sample, seed

seed(42)
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
    # same_hotel_rids = sampled[hid]
    same_hotel_rids = r2d2_features.get_room_ids(hid)
    same_hotel_feature_lists = []
    for same_rid in same_hotel_rids:
        same_hotel_feature_lists.append(r2d2_features.open_and_extract_feature_file(hid, same_rid))
    
    # Compute the cosine similarities between the query image and sampled images from the same hotel
    same_hotel_matches = []
    for same_hotel_features in tqdm(same_hotel_feature_lists, total=len(same_hotel_feature_lists), desc='Computing Closest in Same Hotel'):
    # print(closest_vector_handler(query_img_features, same_hotel_features, 1)[0])
        same_hotel_matches.extend([1-i[0] for i in closest_vector_handler(query_img_features, same_hotel_features, 1)])

    # DIFFERENT HOTEL
    # Get a sample of images from a different hotel
    # Initially, let's just do this for a single different hotel... could wrap in loop for a sample of the total hotels too though

    # diff_hotel_rids = sampled.pop(hid)
    diff_hotel_hids = []
    for id in r2d2_features.get_hotel_ids():
        if id != hid:
            diff_hotel_hids.append(id)
    diff_hotel_feature_lists = []
    for idx, diff_hid in enumerate(list(sample(diff_hotel_hids, 100))):
        ff = r2d2_features.open_and_extract_feature_file(diff_hid)
        diff_rid = int(sample(ff.keys(), 1)[0])
        diff_hotel_feature_lists.append(ff[diff_rid])
        #diff_hotel_feature_lists.extend(list(ff.values())) 
    # Compute the cosine similarities between the query image and sampled images from the diff hotel
    # diff_hotel_feature_lists = list(sample(diff_hotel_features_list, 100)) 
    diff_hotel_matches = []
    for diff_hotel_features in tqdm(diff_hotel_feature_lists, total=len(diff_hotel_feature_lists), desc='Computing Closest in Diff Hotel'):
        diff_hotel_matches.extend([1-i[0] for i in closest_vector_handler(query_img_features, diff_hotel_features, 1)])

    # Save this data
    # Save this data
    array = np.array([same_hotel_matches, diff_hotel_matches])
    np.save('datasets/0.1k/naive_search_results/'+str(hid)+'_'+str(rid)+'.npy', array)
    # print(same_hotel_matches)
    # Plot the distribution of cosine similaritilist(closest_vector_handler(query_img_features, diff_hotel_features, 1)[0])es for each class (hotel)
    #bins = np.linspace(0, 1, 100)
    #ax = plt.gca()
    #ax.set_ylim([0, 10])
    #plt.hist(same_hotel_matches, bins, alpha=0.5, label='same',density=True,stacked=True)
    #plt.hist(diff_hotel_matches, bins, alpha=0.5, label='diff', density=True, stacked=True)
    #plt.legend(loc='upper right')

    #plt.savefig('datasets/0.1k/naive_search_results/'+str(hid)+'_'+str(rid)+'.png')
    #plt.close()
    plot(same_hotel_matches, diff_hotel_matches, hid, rid)
    return same_hotel_matches, diff_hotel_matches

def plot(same_hotel_matches, diff_hotel_matches, hid, rid):
    fig, ax = plt.subplots()
    for a in [(same_hotel_matches, 'same'), (diff_hotel_matches, 'diff')]:
        sns.distplot(a[0],label=a[1], bins=10, ax=ax, kde=True)
    ax.set_xlim([0, 1])
    plt.legend()
    plt.savefig('datasets/0.1k/naive_search_results/'+str(hid)+'_'+str(rid)+'.png')
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

    for hid in sample(hids, 100):
        same, diff = [], []
        for rid in r2d2.get_room_ids(hid):
           s, d = compute_matches_for_image_across_hotels(r2d2, hid, rid)
        same, diff = same.extend(s), diff.extend(d)
        plot(same, diff, hid, 'OVERALL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  parser.add_argument('-d', type=str, required=True) # path to sampled
    parser.add_argument('-p', type=str, required=True) # path to r2d2 obj
    args = parser.parse_args()

    main(args.p)
