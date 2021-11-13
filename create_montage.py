from imutils import build_montages
from imutils import paths
import argparse
import random
import cv2
from util import load_dir,create_dirs, save_image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-s", "--sample", type=int, default=21,
	help="# of images to sample")
args = vars(ap.parse_args())


imagePaths = list(load_dir(args['images'], file_type='png'))
random.shuffle(imagePaths)

# initialize the list of images
images = []
# loop over the list of image paths
for imagePath in imagePaths:
	# load the image and update the list of images
	image = cv2.imread(imagePath)
	images.append(image)
# construct the montages for the images
montages = build_montages(images, (128, 196), (7, 3))

# loop over the montages and display each of them
for montage in montages:
	cv2.imshow("Montage", montage)
	cv2.waitKey(0)

save_path = 'datasets/0.1k/naive_search_results_more_bins/38889/'
create_dirs(save_path)
montages[0].save(save_path+'montage.png')