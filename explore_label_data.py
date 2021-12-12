"""
Now that we have generated all this label data, let's explore it a bit.
Some questions we'd like to answer:
- What is the distribution of c_same?
- What is the distribution of c_diff?
- What is the distribution of their ratio?
- Where do the mean, median, and other quantiles lie for the ratio distribution?
- Are the results consistent with what we would expect, or have we revealed something new?
"""

import matplotlib.pyplot as plt
import numpy as np
from util import open_pckl_file, load_dir

# hotels = open_pckl_file('datasets/0.1k/datasets/same_diff_hotel/38889.pckl')
# hotels = open_pckl_file('datasets/0.1k/datasets/same_diff_hotel/38889.pckl')
# # same/diff, query_points, query_point, (distance, descriptor, xys, hid, iid)
hotel_files = load_dir('./datasets/0.1k/datasets/same_diff_hotel')
total_good = 0
total_points = 0
for hotel in hotel_files:
    hotel = open_pckl_file(hotel)
    good_per_hotel = 0
    for i in range(len(hotel[1])):
        if hotel[1][i][0] / hotel[2][i][0] < 1:
            good_per_hotel += 1
    total_good += good_per_hotel
    total_points += len(hotel[1])
    
print("Percent total points that are good  --- " + str(total_good / total_points))
