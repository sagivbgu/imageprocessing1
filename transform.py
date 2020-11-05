from translate import *
import numpy as np
import cv2 as cv
from math import cos, sin
from functools import reduce
from math import floor

def transformation_to_matrices(trans_gen):
    return multiple_matrices(create_matrices(trans_gen))


def create_matrices(trans_gen):
    """
    Get a generator of tuples from the trans_file
    and create the matrix to apply on the image
    :param trans_gen: generator received from load_trans_file
    :return: one matrix to rule them all
    """
    matrices = []

    for item in trans_gen:
        command, x, y = item
        x = float(x)
        y = float(y)

        if command == "S":
            m = create_scale_matrix(x, y)
        elif command == "R":
            m = create_rotate_matrix(x)
        elif command == "T":
            m = create_translate_matrix(x, y)

        matrices.append(m)

    return matrices


def multiple_matrices(mats):
    if len(mats) > 1:
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


def create_rotate_matrix(theta):
    """
    :param theta: the angle to rotate
    :return: np Mat for rotation
    """
    return np.float32([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])


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

    new_img = create_empty_img(new_height, new_width)
    print(new_img.shape)
    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y, _ = final_mat.dot(np.float32([x, y, 1]))
            new_x = int(floor(new_x))
            new_y = int(floor(new_y))
            new_img[new_x, new_y] = img[x, y]

    return new_img


def determine_new_boundaries(final_mat, img):
    height, width = img.shape

    max_height = int(floor(max([
        final_mat.dot(np.float32([0, 0, 1]))[0],
        final_mat.dot(np.float32([0, width - 1, 1]))[0],
        final_mat.dot(np.float32([height - 1, 0, 1]))[0],
        final_mat.dot(np.float32([height - 1, width - 1, 1]))[0]]))) + 1

    max_width = int(floor(max([
        final_mat.dot(np.float32([0, 0, 1]))[1],
        final_mat.dot(np.float32([0, width - 1, 1]))[1],
        final_mat.dot(np.float32([height - 1, 0, 1]))[1],
        final_mat.dot(np.float32([height - 1, width - 1, 1]))[1]]))) + 1

    return max_height, max_width


def create_empty_img(h, w):
    return np.zeros(shape=[h, w], dtype=np.uint8)


def inverse_mat(mat):
    return np.linalg.inv(mat)


def apply_trans_on_img(trans, img):
    final_mat = transformation_to_matrices(trans)
    new_img = apply_geo_matrix_on_image(final_mat, img)

    return new_img, final_mat, inverse_mat(final_mat)