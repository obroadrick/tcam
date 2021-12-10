import numpy as np

hotels = np.load('datasets/0.1k/datasets/same_diff_hotel/38889.npy', allow_pickle=True)[0]

# # same/diff, query_points, query_point, (distance, descriptor, xys, hid, iid)

good = 0

for i in range(len(hotels[0])):
    if hotels[0][i][0] / hotels[1][i][0] < 1:
        good += 1
        
print("Percent total points that are good from hotel_id=38889 --- " + str(good / len(hotels[0])))