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
    # Calculate old and new size of the images
    old_height, old_width = img.shape
    new_height, new_width = determine_new_boundaries(final_mat, img)

    # Create a new fitting image; all pixels are set to WHITE
    new_img = create_empty_img(new_height + 1, new_width + 1)

    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y = calc_coordinates(final_mat, x, y)
            new_x = round(new_x)
            new_y = round(new_y)
            if not does_exceed(new_x, new_y, new_height, new_width):
                new_img[new_y, new_x] = img[y, x]

    return new_img


def create_matrices(trans_gen, img):
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

    m = multiple_matrices([t1, r, t2])

    _cos = np.abs(m[0, 0])
    _sin = np.abs(m[0, 1])

    new_width = int((height * _sin) + (width * _cos))
    new_height = int((height * _cos) + (width * _sin))

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
    height, width = img.shape

    tl_x, tl_y = calc_coordinates(final_mat, 0, 0)
    tr_x, tr_y = calc_coordinates(final_mat, width - 1, 0)
    bl_x, bl_y = calc_coordinates(final_mat, 0, height - 1)
    br_x, br_y = calc_coordinates(final_mat, width - 1, height - 1)

    max_width = round(max(tl_x, tr_x, bl_x, br_x))
    max_height = round(max(tl_y, tr_y, bl_y, br_y))

    return max_height, max_width


def create_empty_img(h, w):
    # All White
    return 255 + np.zeros(shape=[h, w], dtype=np.uint8)


def multiple_matrices(mats):
    if len(mats) > 1:
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
