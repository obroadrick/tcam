import argparse
import random
from random import sample

from calculate_metadata import get_data
from util import open_pckl_file, write_pckl_5_file

random.seed(42)

def main(factor):
    r2d2 = open_pckl_file('./r2d2_features.pckl')
    hids = r2d2.get_hotel_ids()
    _, __, total = get_data(hids, r2d2)

    new_rooms = {}
    for h in hids:
        rids = r2d2.get_room_ids(h)

        sample_number = len(rids) // factor

        sample_number = max(min(len(rids), 10), sample_number)

        new_rooms[h] = sample(rids, sample_number)

    write_pckl_5_file('./datasets/0.1k/sampled_hotels/sampled_data_{}_factor.pckl'.format(factor), new_rooms)
    print(total, sum([len(i) for i in new_rooms.values()]))
    print("Compression Ratio: {}".format(total / sum([len(i) for i in new_rooms.values()])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True)
    args = parser.parse_args()
    factor = int(args.f)
    main(factor=factor)
