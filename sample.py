import random
import math
from calculate_metadata import get_data
from util import open_pckl_file, write_pckl_5_file
from random import sample

random.seed(42)

r2d2 = open_pckl_file('./r2d2_features.pckl')

hids = r2d2.get_hotel_ids()
min_num, max_num, total = get_data(hids, r2d2)

new_rooms = {}
factor = 2
for h in hids:
    rids = r2d2.get_room_ids(h)
    new_rooms[h] = random.sample(rids, len(rids) // factor)

write_pckl_5_file('./sampled_data.pckl', new_rooms)
print(total, sum([len(i) for i in new_rooms.values()]))