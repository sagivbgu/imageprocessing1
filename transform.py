import numpy as np
from math import cos, sin, radians
from functools import reduce

def apply_trans_on_img(trans, img):
    final_mat = transformation_to_matrices(trans, img)
    new_img = apply_geo_matrix_on_image(final_mat, img)

    return new_img, final_mat, inverse_mat(final_mat)


def transformation_to_matrices(trans_gen, img):
    return multiple_matrices(create_matrices(trans_gen, img))


# TODO: fix problem when rotating - do not cut the picture
def apply_geo_matrix_on_image(final_mat, img):
    old_height, old_width = img.shape
    new_height, new_width = determine_new_boundaries(final_mat, img)

    new_img = create_empty_img(new_height + 1, new_width + 1)

    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y = calc_coordinates(final_mat, x, y)
            new_x = round(new_x)
            new_y = round(new_y)
            if not does_exceed(new_x, new_y, new_height, new_width):
                new_img[new_x, new_y] = img[x, y]

    return new_img


def create_matrices(trans_gen, img):
    # First we need to determine the center for rotation
    center_x, center_y = calc_center(img)
    center_x = float(center_x)
    center_y = float(center_y)

    matrices = []

    for item in trans_gen:
        command, x, y = item
        x = float(x)
        y = float(y)

        if command == "S":
            m = create_scale_matrix(x, y)
        elif command == "R":
            m = create_rotate_matrix(x, center_x, center_y)
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


def create_rotate_matrix(theta, center_x, center_y):
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

    return m


def create_translate_matrix(x, y):
    return np.float32([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])


def determine_new_boundaries(final_mat, img, to_shift=False):
    height, width = img.shape

    tl = final_mat.dot(np.float32([0, 0, 1]))
    tr = final_mat.dot(np.float32([0, width - 1, 1]))
    bl = final_mat.dot(np.float32([height - 1, 0, 1]))
    br = final_mat.dot(np.float32([height - 1, width - 1, 1]))

    max_height = round(max(tl[0], tr[0], bl[0], br[0]))
    min_height = round(min(tl[0], tr[0], bl[0], br[0]))

    max_width = round(max(tl[1], tr[1], bl[1], br[1]))
    min_width = round(min(tl[1], tr[1], bl[1], br[1]))

    new_h = max_height
    new_w = max_width

    if to_shift:
        delta_h = abs(max_height - min_height)
        delta_w = abs(max_width - min_width)
        new_h = max_height + delta_h
        new_w = max_width + delta_w

    return new_h, new_w


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
