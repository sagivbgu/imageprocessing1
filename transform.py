from translate import *
import numpy as np
import cv2 as cv
from math import cos, sin
from functools import reduce


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
    return reduce(np.dot, mats)


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
        [1, 0, 0],
        [0, 1, 0],
        [x, y, 1]
    ])

