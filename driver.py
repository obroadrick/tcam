import os
from random import choice, sample

from util import create_dirs, open_pckl_file, save_image, save_to_csv
from closeness_computations import closest_vector_handler, remove_overflow_points
from visualization import draw_closest_pairs_on_two_images
from PIL import Image

r2d2_features = open_pckl_file('r2d2_features.pckl')

def closest_n_vectors(h1, h2, i1, i2, n):
    """ 
    Takes dicts for descriptions, points, scores (h1 and h2) and paths to 
    images (i1 and i2) and n (how many closest vectors to consider),
    and computes and draws on a new image a visualization of the pairs of points.
    """
    i1 = Image.open(i1)
    i2 = Image.open(i2)

    # Exclude points where radius is over border of image.
    # h1['xys'], h2['xys'], h1['desc'], h2['desc'] = remove_overflow_points(
    #     h1['xys'], h1['desc'], i1.size, h2['xys'], h2['desc'], i2.size)

    # Find closest from each  h1 to closest in h2.
    n_smallest = closest_vector_handler(h1, h2, 10)

    # Draw a nice picture of these results
    lines = False
    circles = False
    image = draw_closest_pairs_on_two_images(i1, i2, n_smallest, lines, circles)

    # Save.
    return image, n_smallest
 
def compare_two(hotel_ids, room_ids):
    h1 = r2d2_features.open_and_extract_feature_file(hotel_ids[0], room_ids[0])
    h2 = r2d2_features.open_and_extract_feature_file(hotel_ids[1], room_ids[1])

    i1 = r2d2_features.get_path_to_image(hotel_ids[0], room_ids[0])
    i2 = r2d2_features.get_path_to_image(hotel_ids[1], room_ids[1])

    stitched_image, n_smallest = closest_n_vectors(h1, h2, i1, i2, 25)

    save(hotel_ids, room_ids, stitched_image, n_smallest)

def save(hotel_ids, room_ids, stitched_image, n_smallest):
    fn = str(hotel_ids[0])+'_'+str(room_ids[0]) +'_'+str(hotel_ids[1])+'_'+str(room_ids[1])
    dir = '.' + os.path.basename(fn+'edge points')
    create_dirs(dir)

    save_image(dir, fn, stitched_image)
    save_to_csv(dir, fn, n_smallest)


def get_random_hotel_rooms(n=2):
    hotel_ids = sample(r2d2_features.get_hotel_ids(), 2)
    room_ids = [sample(r2d2_features.get_room_ids(hotel_ids[0]), 1)[0], sample(
        r2d2_features.get_room_ids(hotel_ids[1]), 1)[0]]

    return hotel_ids, room_ids

def get_same_hotels(hotel_id=None, n=2):
    if hotel_id is None:
        hotel_id = choice(r2d2_features.get_hotel_ids())

    hotel_ids = [hotel_id for i in range(n)]

    room_ids = sample(r2d2_features.get_room_ids(hotel_ids[0]), n)
    return hotel_ids, room_ids

# @jit
def get_diff_hotels(hotel_id):
    hotel_ids = []
    room_ids = []
    for hid in r2d2_features.get_hotel_ids():
        rids = r2d2_features.get_room_ids(hid)
        room_ids.extend(rids)
        hotel_ids.extend([hid for i in range(len(rids))])
    return hotel_ids, room_ids


def main():

    # compare random two
    # hotel_ids, room_ids = get_random_hotel_rooms()
    hotel_ids = ['6268', '6268']
    room_ids = [7891968, 2618161]
    compare_two(hotel_ids, room_ids)
    """
    hotel_ids, room_ids = get_same_hotels()
    compare_two(hotel_ids, room_ids)
    """
    # getting a bunch of random same hotel comparisons
    # n = 10
    # for i in range(n):
    #     try:
    #         hotel_ids, room_ids = get_same_hotels()
    #         compare_two(hotel_ids, room_ids)
    #     except: continue #bad code`
    # end compare random two

    # TODO:get closest in same and different hotels. plot distribution

if __name__ == '__main__':
    main()
