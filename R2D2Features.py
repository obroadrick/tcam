import os
import re
from util import load_dir, open_pckl_5_file


class R2D2Features:
    def __init__(self, features_path, h2i_path, hotel_info_path) -> None:
        # protected, no need for it to be changed later
        self._feature_files = load_dir(features_path)
        self._hotel_info = open_pckl_5_file(hotel_info_path)
        self._hotel_ids = [os.path.basename(i).replace(
            '.pkl', '') for i in self._feature_files]

        h2i = open_pckl_5_file(h2i_path)
        self._room_ids_2_path = {}
        pat = r'.*\/([^_]*)_.*'

        for key in h2i:
            for image in h2i[key]:
                match = re.search(pat, image['img'])
                self._room_ids_2_path[match.group(1)] = image['img'].replace(
                    '/lab/vislab/DATA/ActualFullTraffickCam/', '/pless_nfs/home/datasets/FullTraffickcam/')

    def get_room_ids(self, hotel_id):
        return list(self.open_and_extract_feature_file(hotel_id).keys())

    def get_hotel_ids(self):
        return self._hotel_ids

    def open_and_extract_feature_file(self, hotel_id, room_id=None):
        index = [idx for idx, s in enumerate(
            self._feature_files) if str(hotel_id) in s][0]
        hotel_feature_file = self._feature_files[index]
        if room_id is not None:
            hotel_feature_data = open_pckl_5_file(hotel_feature_file)[room_id]
        else:
            hotel_feature_data = open_pckl_5_file(hotel_feature_file)

        return hotel_feature_data

    def get_path_to_image(self, hotel_id, room_id):
        # path_to_image = self._h2i[str(hotel_id)]
        # path_to_image = [i['img'] for i in path_to_image if str(room_id) in i['img']][0].replace(
        #     '/lab/vislab/DATA/ActualFullTraffickCam/', '/pless_nfs/home/datasets/FullTraffickcam/')
        path_to_image = self._room_ids_2_path[str(room_id)]
        return path_to_image


if __name__ == '__main__':
    features_path = "/pless_nfs/home/datasets/tcam_local_features/small"
    h2i_path = './tcam_for_gwu/hotel2img_dict.pkl'
    h_info_path = './tcam_for_gwu/hotelimageinfo.pkl'

    r2d2_features = R2D2Features(features_path, h2i_path, h_info_path)
