import numpy as np
from math import cos, sin, radians
from functools import reduce


def transformation_to_matrices(trans_gen, img):
    return multiple_matrices(create_matrices(trans_gen, img))


def create_matrices(trans_gen, img):
    """
    Get a generator of tuples from the trans_file
    and create the matrix to apply on the image
    :param trans_gen: generator received from load_trans_file
    :return: one matrix to rule them all
    """

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


def multiple_matrices(mats):
    if len(mats) > 1:
        # mats.reverse()
        return reduce(np.dot, mats)
    else:
        return mats[0]


def create_scale_matrix(x, y):
    """
    :param x: scale for x
    :param y: scale for y
    :return: np Mat for scaling
    """
    return np.float32([
        [x, 0, 0],
        [0, y, 0],
        [0, 0, 1]
    ])


def create_rotate_matrix(theta, center_x, center_y):
    """
    :param theta: the angle to rotate
    :return: np Mat for rotation
    """

    # First build translates matrix to center
    t1 = create_translate_matrix(center_x, center_y)
    t2 = create_translate_matrix(-center_x, -center_y)

    r = np.float32([
        [cos(radians(theta)), sin(radians(theta)), 0],
        [-sin(radians(theta)), cos(radians(theta)), 0],
        [0, 0, 1]
    ])

    m = multiple_matrices([t1, r, t2])
    print(m)
    return m


def create_translate_matrix(x, y):
    """
    :param x: trans for x
    :param y: trans for y
    :return: np Mat for translation
    """
    return np.float32([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])


def apply_geo_matrix_on_image(final_mat, img):
    old_height, old_width = img.shape
    new_height, new_width = determine_new_boundaries(final_mat, img)

    new_img = create_empty_img(new_height + 1, new_width + 1)

    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y = calc_coordinates(final_mat, x, y)
            new_x = int(round(new_x))
            new_y = int(round(new_y))
            if not does_exceed(new_x, new_y, new_height, new_width):
                new_img[new_x, new_y] = img[x, y]

    return new_img


def determine_new_boundaries(final_mat, img):
    height, width = img.shape

    max_height = int(round(max([
        final_mat.dot(np.float32([0, 0, 1]))[0],
        final_mat.dot(np.float32([0, width - 1, 1]))[0],
        final_mat.dot(np.float32([height - 1, 0, 1]))[0],
        final_mat.dot(np.float32([height - 1, width - 1, 1]))[0]])))

    max_width = int(round(max([
        final_mat.dot(np.float32([0, 0, 1]))[1],
        final_mat.dot(np.float32([0, width - 1, 1]))[1],
        final_mat.dot(np.float32([height - 1, 0, 1]))[1],
        final_mat.dot(np.float32([height - 1, width - 1, 1]))[1]])))

    return max_height, max_width


def create_empty_img(h, w):
    print(h)
    print(w)
    return np.zeros(shape=[h, w], dtype=np.uint8)


def inverse_mat(mat):
    return np.linalg.inv(mat)


def apply_trans_on_img(trans, img):
    final_mat = transformation_to_matrices(trans, img)
    new_img = apply_geo_matrix_on_image(final_mat, img)

    return new_img, final_mat, inverse_mat(final_mat)


def does_exceed(x, y, h, w):
    return x < 0 or y < 0 or x > w - 1 or y > h - 1


def calc_center(img):
    h, w = img.shape
    return w / 2, h / 2


def calc_coordinates(mat, x, y):
    new_x, new_y, _ = mat.dot(np.float32([x, y, 1]))
    return new_x, new_y
