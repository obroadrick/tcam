from R2D2Features import R2D2Features
from util import write_pckl_file
import os
import argparse

def main(path, h2i, h_info, save_path, fn):
    r2d2_features = R2D2Features(path, h2i, h_info)
    write_pckl_file(os.path.join(save_path, fn), r2d2_features)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True)
    parser.add_argument('-s', type=str, required=True)
    parser.add_argument('--fn', type=str, required=True)

    args = parser.parse_args()

    features_path = args.p
    h2i_path = './tcam_for_gwu/hotel2img_dict.pkl'
    h_info_path = './tcam_for_gwu/hotelimageinfo.pkl'

    main(features_path, h2i_path, h_info_path, args.s, args.fn)
