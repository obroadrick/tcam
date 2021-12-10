from util import open_pckl_5_file
from tqdm import tqdm

r2d2 = open_pckl_5_file('datasets/0.1k/r2d2_objs/clean.pckl')

hids = r2d2.get_hotel_ids()

num_points = 0
num_images = 0

for hid in tqdm(hids):
    ff = r2d2.open_and_extract_feature_file(hid)
    num_images += len(list(ff.keys()))
    
    for rid in ff.keys():
        num_points += len(ff[rid]['scores'])

print(num_points)
print(num_images)
print(len(hids))
with open('./datasets/0.1k/stats.txt', 'w+') as f:
    f.write('num_points={}\n'.format(num_points))
    f.write('num_images={}\n'.format(num_images))
    f.write('num_hotels={}\n'.format(len(hids)))


