"""
Functions for visualizing r2d2 points on images (and pairs of close r2d2 points across multiples images).
"""
import math
import numpy as np
import copy
from PIL import Image, ImageDraw
from random import randint


def draw_closest_pairs_on_two_images(i1, i2, n_smallest, lines=True, circles=True):
     # Stitch images together, updating the n_smallest coordinates.
    image, n_smallest = stitch_images(i1, i2, n_smallest)
    print("Found {} nearest matching r2d2 points, excluding those along the image border".format(len(n_smallest)))

    # Draw points in pairs with random matching colors.
    image, colors = draw_points(image, n_smallest)

    if lines:
        # Draw lines between matching points with same matching colors.
        image = draw_lines(image, n_smallest, colors)

    if circles:
        # Draw circles with correct radius for each r2d2 point.
        image = draw_circles(image, n_smallest, colors)
    
    # Return the stiched, drawn upon image
    return image

def closest_same_closest_diff(same_hotels, same_rooms, diff_hotels, diff_rooms):
    pass

def draw_points(image, n_smallest, one_image=False):
    draw = ImageDraw.Draw(image)

    r = 5
    colors = []

    if not one_image:
        for s in n_smallest:
            point1, point2 = s[1][:2], s[2][:2]
            color = (randint(0, 255), randint(0,255), randint(0,255))
            circle1 = (point1[0]-r, point1[1]-r ,point1[0]+r, point1[1]+r)
            circle2 = (point2[0]-r, point2[1]-r, point2[0]+r, point2[1]+r)

            draw.ellipse(circle1, fill=color)
            draw.ellipse(circle2, fill=color)

            colors.append(color)
    else:
        r = 1
        for s in n_smallest[:100]:
            point1 = s
            color = (randint(0, 255), randint(0,255), randint(0,255))
            circle1 = (point1[0]-r, point1[1]-r ,point1[0]+r, point1[1]+r)

            draw.ellipse(circle1, fill=color)

            colors.append(color)

    return image, colors

def draw_lines(image, n_smallest, colors):
    draw = ImageDraw.Draw(image)

    assert len(colors) == len(n_smallest)

    for i in range(len(colors)):
        s = n_smallest[i]
        color = colors[i]
        point1, point2 = s[1][:2], s[2][:2]
        draw.line((point1[0], point1[1], point2[0], point2[1]), fill=color, width=3)

    return image

def draw_circles(image, n_smallest, colors):
    draw = ImageDraw.Draw(image)

    assert len(colors) == len(n_smallest)

    for i in range(len(colors)):
        s = n_smallest[i]
        color = colors[i]

        point1, point2 = s[1][:2], s[2][:2]
        r1, r2 = s[1][2]/2, s[2][2]/2
        circle1 = (point1[0]-r1, point1[1]-r1 ,point1[0]+r1, point1[1]+r1)
        circle2 = (point2[0]-r2, point2[1]-r2, point2[0]+r2, point2[1]+r2)

        draw.ellipse(circle1, fill=None)
        draw.ellipse(circle2, fill=None)

    return image

def stitch_images(image1, image2, nsmallest):
    """
    Resizes the image of smaller height to match the heights of the images.
    Stitches the two images together, updating all the coordinates in h1 and h2.
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size
    nsmallest_stitched = copy.deepcopy(nsmallest)

    # Resize the image with lesser height to match the other image (for viewing pleasure)
    if height1 < height2:
        # We need to scale up image1.
        new_h = height2
        proportion = new_h / height1
        new_w = int(width1 * proportion)
        image1 = image1.resize((new_w,new_h))
        width1 = new_w
        height1 = new_h
        # Update coordinates for image1. 
        # (first index 1 is to get the first im, second index 1 is to get y coord)
        for item in nsmallest_stitched:
            item[1][1] *= proportion
            item[1][0] *= proportion
    elif height2 < height1:
        # We need to scale up image2.
        new_h = height1
        proportion = new_h / height2
        new_w = int(width2 * proportion)
        image2 = image2.resize((new_w,new_h))
        width2 = new_w
        height2 = new_h
        # Update coordinates for points in image 2.
        for item in nsmallest_stitched:
            item[2][1] *= proportion
            item[2][0] *= proportion

    # Update x coordinates for points in image2.
    for item in nsmallest_stitched:
        item[2][0] += width1

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result, nsmallest_stitched
