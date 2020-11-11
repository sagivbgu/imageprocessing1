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
    new_img, final_mat = apply_geo_matrix_on_image(final_mat, img)
    return new_img, final_mat, inverse_mat(final_mat)


def transformation_to_matrices(trans_gen, img):
    return multiple_matrices(create_matrices(trans_gen, img))


def apply_geo_matrix_on_image(final_mat, img):
    """
    Takes a geometric transformation matrix and transform each pixel from img
    to its new location in the new image
    :param final_mat: the geometric transformation matrix
    :param img: the original image
    :return: the new image and the final mat (in case it was changed)
    """
    # Calculate old and new size of the images
    old_height, old_width = img.shape
    new_height, new_width, final_mat = determine_new_boundaries_and_fix_negative_translation(final_mat, img)

    # Create a new fitting image; all pixels are set to WHITE
    new_img = create_empty_img(new_height, new_width)

    # copy old pixels to their new position
    for y in range(old_height):
        for x in range(old_width):
            new_x, new_y = calc_coordinates(final_mat, x, y)
            new_x = round(new_x)
            new_y = round(new_y)
            if not does_exceed(new_x, new_y, new_height, new_width):  # TODO: Remove?
                new_img[new_y, new_x] = img[y, x]

    return new_img, final_mat


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
            m = create_rotate_matrix(x)
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


def create_rotate_matrix(theta):
    # The rotation matrix around (0,0)
    return np.float32([
        [cos(radians(theta)), sin(radians(theta)), 0],
        [-sin(radians(theta)), cos(radians(theta)), 0],
        [0, 0, 1]
    ])


def create_translate_matrix(x, y):
    return np.float32([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])


def determine_new_boundaries_and_fix_negative_translation(final_mat, img):
    """
    Given an original image and a geometric transformation matrix, this function
    calculates the size of the new image.
    If pixels are translated to negative coordinates (upwards or to the left), the function adjusts the transformation
    so that these pixels will remain in the new image, but the size of the image will be increased downwards and to the
    right. The result is that no pixel is "cut off" the image, and boundaries are expanded, creating an "illusion"
    of translating upwards or to the left
    :param final_mat: the final geo transformation as composed from the given trans file
    :param img: the original image
    :return: the new image boundaries, and the fixed transformation matrix
    """
    # Calculate the new height and width of the new image
    # Based on similar triangles trigonometric
    height, width = img.shape

    new_width = width
    new_height = height

    # get the sizes of the current translation
    trans_on_x = final_mat[0, 2]
    trans_on_y = final_mat[1, 2]

    # Get the coordinates of the corners
    tl_x, tl_y = calc_coordinates(final_mat, 0, 0)
    tr_x, tr_y = calc_coordinates(final_mat, width, 0)
    bl_x, bl_y = calc_coordinates(final_mat, 0, height)
    br_x, br_y = calc_coordinates(final_mat, width, height)

    # calculate the minimum values of each
    min_width = round(min(tl_x, tr_x, bl_x, br_x))
    min_height = round(min(tl_y, tr_y, bl_y, br_y))

    # == Fixing negative translation
    # Now calculate the most significant NEGATIVE translation on X axis
    if min_width < 0:
        min_width = round(abs(min_width))
        max_trans_on_x = max(round(abs(trans_on_x)), min_width)
        # Now we know the real translation on X, so adjust the size and the matrix
        new_width += max_trans_on_x
        final_mat[0, 2] += max_trans_on_x
    else:
        # no pixel is translated to negative X coordinate, so just adjust the width
        new_width += round(abs(trans_on_x))

    # The same, for Y axis
    if min_height < 0:
        min_height = round(abs(min_height))
        max_trans_on_y = max(round(abs(trans_on_y)), min_height)
        new_height += max_trans_on_y
        final_mat[1, 2] += max_trans_on_y
    else:
        new_height += round(abs(trans_on_y))

    # == Check that now we are not out of bounds on the positive directions (downwards and to the right)
    # Get the NEW coordinates of the corners
    tl_x, tl_y = calc_coordinates(final_mat, 0, 0)
    tr_x, tr_y = calc_coordinates(final_mat, width - 1, 0)
    bl_x, bl_y = calc_coordinates(final_mat, 0, height - 1)
    br_x, br_y = calc_coordinates(final_mat, width - 1, height - 1)

    # calculate the minimum values of each
    max_width = round(max(tl_x, tr_x, bl_x, br_x))
    max_height = round(max(tl_y, tr_y, bl_y, br_y))

    # If the pixels are exceeding
    if max_width > new_width:
        # adjust the new size of the image
        delta_w = round(max_width - new_width)
        new_width += delta_w

    if max_height > new_height:
        delta_h = round(max_height - new_height)
        new_height += delta_h

    return new_height, new_width, final_mat


def create_empty_img(h, w, color=255):
    """
    Create a matrix representing an empty image
    :param h: The desired height
    :param w: The desired width
    :param color: The color to fill the matrix with
    """
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
    """
    Check whether a pixel (x, y) is outside the boundaries of a matrix with dimensions height * width
    :param x: The pixel's x coordinate
    :param y: The pixel's y coordinate
    :param h: The matrix's height
    :param w: The matrix's width
    """
    return x < 0 or y < 0 or x > w - 1 or y > h - 1


def calc_center(img):
    """
    Get the (x, y) positions of the pixel at the center of the image
    :param img: The image
    """
    h, w = img.shape
    return w / 2, h / 2


def calc_coordinates(mat, x, y):
    """
    Apply the transformation matrix to a pixel (x, y)
    :param mat: The transformation matrix
    :param x: The pixel's x coordinate
    :param y: The pixel's y coordinate
    """
    new_x, new_y, _ = mat.dot(np.float32([x, y, 1]))
    return new_x, new_y
