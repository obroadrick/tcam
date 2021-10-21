from R2D2Features import R2D2Features
import pickle

features_path = "./datasets/cleaned_0.1k/"
h2i_path = './tcam_for_gwu/hotel2img_dict.pkl'
h_info_path = './tcam_for_gwu/hotelimageinfo.pkl'

r2d2_features = R2D2Features(features_path, h2i_path, h_info_path)

with open('cleaned_r2d2.pckl', 'wb') as f:
    pickle.dump(r2d2_features, f)