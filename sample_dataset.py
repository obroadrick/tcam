""" Sample and label """
from os import write
import pandas as pd
import numpy as np
import random

from util import open_pckl_5_file, write_pckl_file
from tqdm import tqdm

random.seed(42)

R2D2 = open_pckl_5_file('datasets/0.1k/r2d2_objs/clean.pckl')

hotel_ids = R2D2.get_hotel_ids()

# Sample 10 random hotels uniformly
hotel_ids = random.sample(hotel_ids, 10)

# Sample *all* images (let's not miss the painting or nice angle of lamp in one of them)
proportion_of_points_to_sample = .25
sample = {}
for hid in tqdm(hotel_ids):
    image_ids = R2D2.get_room_ids(hid) # these are actually image id's :) research code!
    feature_file = R2D2.open_and_extract_feature_file(hid)

    hotel_pts = {}
    for im_id in image_ids:
        # Sample uniformly .25 of all points per image (which gives a sample of .25 of the points per hotel)
        points = list(zip(feature_file[im_id]['desc'], feature_file[im_id]['xys'][:, :2]))
        
        random.seed(42)
        sampled_points = random.sample(points,  int(proportion_of_points_to_sample * len(points)))
        hotel_pts[im_id] = sampled_points

    sample[hid] = hotel_pts
    
write_pckl_file('./datasets/0.1k/datasets/10_hotel_sample.pckl', sample)         

# Make sure that all our data is manageable (#TODO temporary after first phase)












