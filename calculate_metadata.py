from os import write
from tqdm import tqdm
from util import open_pckl_file
def get_data(h_ids, r2d2=None):
    min_num = float('inf')
    max_num = 0
    total = 0

    for h in tqdm(h_ids):
        num_h = len(r2d2.get_room_ids(h))
        min_num = min(min_num, num_h)
        max_num = max(max_num, num_h)
        total += num_h
    return min_num, max_num, total

if __name__ =='__main__':
    filename='./metadata.txt'
    print('Starting Computations')
    r2d2 = open_pckl_file('./r2d2_features.pckl')

    h_ids = r2d2.get_hotel_ids()
    min_num, max_num, total = get_data(h_ids)
    message = "total={}\nmin={}\nmax={}\navg={}".format(total, min_num, max_num, total/len(h_ids))
    print(message)
    with open(filename, 'w+') as f:
        f.write(message)