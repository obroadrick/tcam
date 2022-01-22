import random
from shutil import copyfile

from util import open_pckl_5_file
from tqdm import tqdm
import os

random.seed(42)
R2D2 = open_pckl_5_file('/pless_nfs/home/mdt_/tcam/datasets/0.1k/r2d2_objs/clean.pckl')
hotel_ids = R2D2.get_hotel_ids()

# Sample 10 random hotels uniformly
hotel_ids = random.sample(hotel_ids, 10)

h2i = open_pckl_5_file('/pless_nfs/home/mdt_/tcam/tcam_for_gwu/hotel2img_dict.pkl')

paths = []
for hid in tqdm(hotel_ids):
    for image in h2i[str(hid)]:
        path = image['img'].replace('/lab/vislab/DATA/ActualFullTraffickCam/', '/pless_nfs/home/datasets/FullTraffickcam/')
        copyfile(path, './images/'+os.path.basename(path))




