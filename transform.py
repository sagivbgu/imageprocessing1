import numpy as np
from math import cos, sin, radians
from functools import reduce


def apply_trans_on_img(trans, img):
    """
    Apply all the transformations listed in trans on the given img
    :param trans: list of transformation from the trans_file
    :param img: the image to process
    :return: The geometric transformed image, the transformations matrix and its inverse
    """
    final_mat = transformation_to_matrices(trans, img)
    new_img = apply_geo_matrix_on_image(final_mat, img)

    return new_img, final_mat, inverse_mat(final_mat)


def transformation_to_matrices(trans_gen, img):
    return multiple_matrices(create_matrices(trans_gen, img))


def apply_geo_matrix_on_image(final_mat, img):
    """
    Takes a geometric transformation matrix and transform each pixel from img
    to its new location in the new image
    :param final_mat: the geometric transformation matrix
    :param img: the original image
    :return: the new image
    """
    # Calculate old and new size of the images
    old_height, old_width = img.shape
    new_height, new_width = determine_new_boundaries(final_mat, img)

    # Create a new fitting image; all pixels are set to WHITE
    new_img = create_empty_img(new_height + 1, new_width + 1)

    # copy old pixels to their new position
    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y = calc_coordinates(final_mat, x, y)
            new_x = round(new_x)
            new_y = round(new_y)
            if not does_exceed(new_x, new_y, new_height, new_width):
                new_img[new_y, new_x] = img[y, x]

    return new_img


def create_matrices(trans_gen, img):
    """
    Create the correct matrices according to the list from the file
    :param trans_gen: the list of transformations
    :param img: the original image
    :return: a list of the transformation matrices
    """
    matrices = []

    for item in trans_gen:
        command, x, y = item
        x = float(x)
        y = float(y)

        if command == "S":
            m = create_scale_matrix(x, y)
        elif command == "R":
            m = create_rotate_matrix(x, img)
        elif command == "T":
            m = create_translate_matrix(x, y)
        else:
            raise ValueError  # This shouldn't happen since we assume the transformation file is correct

        matrices.append(m)

    return matrices


def create_scale_matrix(x, y):
    return np.float32([
        [x, 0, 0],
        [0, y, 0],
        [0, 0, 1]
    ])


def create_rotate_matrix(theta, img):
    height, width = img.shape

    # First we need to determine the center for rotation
    center_x, center_y = calc_center(img)
    center_x = float(center_x)
    center_y = float(center_y)

    # First build translates matrices to rotate around the center
    t1 = create_translate_matrix(center_x, center_y)
    t2 = create_translate_matrix(-center_x, -center_y)

    # The rotation matrix
    r = np.float32([
        [cos(radians(theta)), sin(radians(theta)), 0],
        [-sin(radians(theta)), cos(radians(theta)), 0],
        [0, 0, 1]
    ])

    # The rotation matrix around the center
    m = multiple_matrices([t1, r, t2])

    # Now we deal with the cut off at the edges
    # Get the cosine and sine of the angle of rotation
    _cos = np.abs(m[0, 0])
    _sin = np.abs(m[0, 1])

    # Calculate the new height and width
    new_width = int((height * _sin) + (width * _cos))
    new_height = int((height * _cos) + (width * _sin))

    # Add to the matrix a translation to all coordinates
    # Fix the error around the center
    m[0, 2] += (new_width / 2) - center_x
    m[1, 2] += (new_height / 2) - center_y

    return m


def create_translate_matrix(x, y):
    return np.float32([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])


def determine_new_boundaries(final_mat, img):
    """
    Given an original image,
    :param final_mat:
    :param img:
    :return:
    """
    height, width = img.shape

    tl_x, tl_y = calc_coordinates(final_mat, 0, 0)
    tr_x, tr_y = calc_coordinates(final_mat, width - 1, 0)
    bl_x, bl_y = calc_coordinates(final_mat, 0, height - 1)
    br_x, br_y = calc_coordinates(final_mat, width - 1, height - 1)

    max_width = round(max(tl_x, tr_x, bl_x, br_x))
    max_height = round(max(tl_y, tr_y, bl_y, br_y))

    return max_height, max_width


def create_empty_img(h, w, color=255):
    # All White
    return color + np.zeros(shape=[h, w], dtype=np.uint8)


def multiple_matrices(mats):
    if len(mats) > 1:
        mats.reverse()  # (Tn...(T2(T1(img))...)
        return reduce(np.dot, mats)
    else:
        return mats[0]


def inverse_mat(mat):
    return np.linalg.inv(mat)


def does_exceed(x, y, h, w):
    return x < 0 or y < 0 or x > w - 1 or y > h - 1


def calc_center(img):
    h, w = img.shape
    return w / 2, h / 2


def calc_coordinates(mat, x, y):
    new_x, new_y, _ = mat.dot(np.float32([x, y, 1]))
    return new_x, new_y


def add_margins(img, add_h=2, add_w=2):
    h, w = img.shape
    new_h = h + add_h * 2
    new_w = w + add_w * 2

    new_image = create_empty_img(new_h, new_w)
    for y in range(h):
        for x in range(w):
            new_x = x + add_w
            new_y = y + add_h
            new_image[new_y, new_x] = img[y, x]

    return new_image
