import os
import re
from tqdm import tqdm
from util import load_dir, open_pckl_5_file, open_pckl_file, write_pckl_file
from multiprocessing import Process, Manager

class R2D2Features:
    def __init__(self, features_path, h2i_path, hotel_info_path) -> None:
        # protected, no need for it to be changed later
        self._feature_files = load_dir(features_path)

        hids = [int(os.path.basename(f).replace('.pkl', '').replace('.pckl', '')) for f in self._feature_files]

        self.h2i = open_pckl_5_file(h2i_path)

        def open_and_append(d, ids, proc):
            print("in proc {}".format(proc))
            for idx, hid in tqdm(enumerate(ids), total=len(ids)):
                ff = open_pckl_5_file(self._feature_files[idx])
                d[int(hid)] = list(ff.keys())

        self._hotel_ids_2_image_ids = {}
        self._image_ids_2_path = {}
        pat = r'.*\/([^_]*)_.*'

        for key in hids:
            self._hotel_ids_2_image_ids[key] = []
            for image in self.h2i[str(key)]:
                match = re.search(pat, image['img'])
                self._hotel_ids_2_image_ids[int(key)].append(int(match.group(1)))
                self._image_ids_2_path[int(match.group(1))] = image['img'].replace(
                    '/lab/vislab/DATA/ActualFullTraffickCam/', '/pless_nfs/home/datasets/FullTraffickcam/')


    def get_room_ids(self, hotel_id):
        # return list(self.open_and_extract_feature_file(hotel_id).keys())
        return list(self._hotel_ids_2_image_ids[int(hotel_id)])

    def get_hotel_ids(self):
        return list(self._hotel_ids_2_image_ids.keys())

    def open_and_extract_feature_file(self, hotel_id, room_id=None):
        index = [idx for idx, s in enumerate(
            self._feature_files) if str(hotel_id) in s][0]
        hotel_feature_file = self._feature_files[index]
        if room_id is not None:
            hotel_feature_data = open_pckl_5_file(hotel_feature_file)[room_id]
        else:
            hotel_feature_data = open_pckl_5_file(hotel_feature_file)
        # print(hotel_feature_data)
        return hotel_feature_data

    def get_path_to_image(self, hotel_id, room_id):
        # path_to_image = self._h2i[str(hotel_id)]
        # path_to_image = [i['img'] for i in path_to_image if str(room_id) in i['img']][0].replace(
        #     '/lab/vislab/DATA/ActualFullTraffickCam/', '/pless_nfs/home/datasets/FullTraffickcam/')
        path_to_image = self._image_ids_2_path[int(room_id)]
        return path_to_image
