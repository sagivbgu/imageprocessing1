import numpy as np
import math
from transform import calc_coordinates, does_exceed

# def find_nearest_pixel(img, i, j):
#     try:
#         left = img[i][j - 1]
#         if is_pixel_blank(left):
#             return left
#     except IndexError:
#         pass
#     try:
#         top = img[i - 1][j]
#         if is_pixel_blank(top):
#             return top
#     except IndexError:
#         return img[i][j]
#

def interpolation_nearest(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape

    for new_i in range(new_rows):
        for new_j in range(new_cols):
            old_i, old_j = calc_coordinates(inverse_transformation, new_i, new_j)
            old_i = round(old_i)
            old_j = round(old_j)
            if not (does_exceed(old_i, old_j, old_rows, old_cols)):
                new_img[new_i][new_j] = original_img[old_i][old_j]


def fract(num):
    return num - math.floor(num)


def interpolation_bilinear(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape
    for new_i in range(new_rows):
        for new_j in range(new_cols):
            old_i, old_j = calc_coordinates(inverse_transformation, new_i, new_j)
            top_i = 0 if round(old_i) == 0 else round(old_i) - 1
            bottom_i = top_i if top_i == old_rows - 1 else top_i + 1
            left_j = 0 if round(old_j) == 0 else round(old_j) - 1
            right_j = left_j if left_j == old_cols - 1 else left_j + 1
            width = math.fabs(old_j - left_j - 0.5)
            height = math.fabs(old_i - bottom_i - 0.5)

            intensity_top = (1 - width) * original_img[top_i][left_j] + width * original_img[top_i][right_j]
            intensity_bottom = (1 - width) * original_img[bottom_i][left_j] + width * original_img[bottom_i][right_j]
            new_img[new_i][new_j] = (1 - height) * intensity_bottom + height * intensity_top


# TODO: Remove prints and tuple()s
def u(d):
    if math.fabs(d) < 1:
        return 1.5 * (d ** 3) - 2.5 * (d ** 2) + 1
    elif math.fabs(d) < 2:
        return -0.5 * (d ** 3) + 2.5 * (d ** 2) - 4 * d + 2
    return 0


def get_cubic_matrix(start_i, start_j, d_i, d_j):
    return tuple((tuple((u(math.fabs(i + d_i)) * u(math.fabs(j + d_j)) for j in range(start_j, start_j + 4)))
                  for i in range(start_i, start_i + 4)))


cubic_tl_matrix = get_cubic_matrix(-2, -2, 0.25, 0.25)
cubic_tr_matrix = get_cubic_matrix(-2, -1, 0.25, -0.25)
cubic_bl_matrix = get_cubic_matrix(-1, -2, -0.25, 0.25)
cubic_br_matrix = get_cubic_matrix(-1, -1, -0.25, -0.25)


def get_matrix(roi):
    if roi == (-2, -2):
        return cubic_tl_matrix
    if roi == (-2, -1):
        return cubic_tr_matrix
    if roi == (-1, -2):
        return cubic_bl_matrix
    if roi == (-1, -1):
        return cubic_br_matrix


def get_roi_range(fract_i, fract_j):
    if fract_i < 0.5 and fract_j < 0.5:
        return -2, -2  # tl
    elif fract_i < 0.5 and fract_j >= 0.5:
        return -2, -1  # tr
    elif fract_i >= 0.5 and fract_j < 0.5:
        return -1, -2  # bl
    else:
        return -1, -1  # br


def interpolation_cubic(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape
    pixels = 0 # TODO: remove, for debugging

    # TODO: pad and shift indexes

    for new_i in range(new_rows):
        for new_j in range(new_cols):
            try:
                old_i, old_j = calc_coordinates(inverse_transformation, new_i, new_j)

                # if the pixel exceeds from the original pic,
                # we don't need to interpolate
                if does_exceed(old_j, old_i, old_rows, old_cols):
                    continue

                roi = get_roi_range(fract(old_i), fract(old_j))
                matrix = get_matrix(roi)
                start_i, start_j = roi
                old_i = math.floor(old_i) # TODO: not sure if this or round
                old_j = math.floor(old_j)

                matrix_roi = original_img[old_i + start_i: old_i + start_i + 4, old_j + start_j: old_j + start_j + 4]
                new_value = calculate_cubic_new_value(matrix_roi, matrix)

                new_img[new_i][new_j] = new_value
                pixels += 1
            except ValueError as e:
                pass
            except Exception as e:
                raise e

    print(pixels)


def calculate_cubic_new_value(mat_roi, mat):
    mat_roi = np.float32(mat_roi).flatten()
    mat = np.float32(mat).flatten()
    return np.inner(mat_roi, mat)
