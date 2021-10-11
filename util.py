import glob
import pickle5 
import pickle
import os
import csv


def open_pckl_5_file(file):
    with open(file, 'rb') as f:
        file_data = pickle5.load(f)
    return file_data

def open_pckl_file(file):
    with open(file, 'rb') as f:
        file_data = pickle.load(f)
    return file_data

def load_dir(path):
    return list(glob.glob(path + "/*.*"))


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_csv(path, fn, data):
    with open(os.path.join(path, fn+'.csv'), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Cos Distance', 'x1y1', 'x2y2'])
        for row in data:
            csv_out.writerow(row)


def save_image(path, fn, image):
    image.save(os.path.join(path, fn+ '.png'))
