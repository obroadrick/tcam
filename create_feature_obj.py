from R2D2Features import R2D2Features
import pickle

features_path = "/pless_nfs/home/datasets/tcam_local_features/small"
h2i_path = './tcam_for_gwu/hotel2img_dict.pkl'
h_info_path = './tcam_for_gwu/hotelimageinfo.pkl'

r2d2_features = R2D2Features(features_path, h2i_path, h_info_path)

with open('r2d2_features.pckl', 'wb') as f:
    pickle.dump(r2d2_features, f)