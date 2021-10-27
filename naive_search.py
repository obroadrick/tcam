"""
Functions for our initial, naive search that we will use to build an 
understanding of how r2d2 points can be used for our matching problem.
"""

from closeness_computations import closest_vector_handler

def compute_matches_for_image_across_hotels(hid, rid):
    """
    This function is the initial "search" algorithm that we are implementing.
    For the query image, we compute the pairwise-closest r2d2 points for a sample 
    of a images from the same hotel and a sample of images from a different hotel.
    We then save and plot the distribution of cosine-similarity for these two classes.

    hid     hotel id of the query image
    rid     room id of the query image
    """
    # Get the query image features
    query_img_features = r2d2_features.open_and_extract_feature_file(hotel_ids[0], room_ids[0])

    # SAME HOTEL
    # Get a sample of images from the same hotel
    same_hotel_rids = #TODO marshall
    same_hotel_feature_lists = []
    for same_rid in same_hotel_rids:
        same_hotel_feature_lists.append(r2d2_features.open_and_extract_feature_file(hid, same_rid))
    
    # Compute the cosine similarities between the query image and sampled images from the same hotel
    same_hotel_matches = []
    for same_hotel_features in same_hotel_feature_lists:
        same_hotel_matches.append(closest_vector_handler(query_img_features, same_hotel_features, 1))

    # DIFFERENT HOTEL
    # Get a sample of images from a different hotel
    # Initially, let's just do this for a single different hotel... could wrap in loop for a sample of the total hotels too though #TODO
    diff_hotel_id = #TODO marshall
    diff_hotel_rids = #TODO marshall
    diff_hotel_feature_lists = []
    for diff_rid in diff_hotel_rids:
        diff_hotel_feature_lists.append(r2d2_features.open_and_extract_feature_file(diff_hotel_hid, diff_rid))
    
    # Compute the cosine similarities between the query image and sampled images from the diff hotel
    diff_hotel_matches = []
    for diff_hotel_features in diff_hotel_feature_lists:
        diff_hotel_matches.append(closest_vector_handler(query_img_features, diff_hotel_features, 1))

    # Save this data
    # TODO decide best way to do so

    # Plot the distribution of cosine similarities for each class (hotel)
    # TODO check how this data is structured so we can quickly plot it