import os

from tqdm import tqdm

from closeness_computations import remove_overflow_points
from util import open_image, open_pckl_file, write_pckl_5_file

R2D2_FEATS = open_pckl_file('./datasets/0.1k/r2d2_objs/unclean.pckl')
hotel_ids = R2D2_FEATS.get_hotel_ids()

dir = './datasets/0.1k/data/'

for hotel in tqdm(hotel_ids):
    feature_file = R2D2_FEATS.open_and_extract_feature_file(hotel_id=hotel)

    for key in list(feature_file.keys()):
        xys, desc, score = feature_file[key]['xys'],feature_file[key]['desc'],feature_file[key]['scores']
        i1 = open_image(R2D2_FEATS.get_path_to_image(str(hotel),key))
        feature_file[key]['xys'], _, feature_file[key]['desc'], _ =  remove_overflow_points(xys, desc, i1.size)
     
    full_path = os.path.join(dir, str(hotel)+'.pckl')
    write_pckl_5_file(full_path,feature_file)
