import numpy as np
import math
from transform import calc_coordinates
from transform import does_exceed


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
    print(d)
    if math.fabs(d) < 1:
        return 1.5 * (d ** 3) - 2.5 * (d ** 2) + 1
    elif math.fabs(d) < 2:
        return -0.5 * (d ** 3) + 2.5 * (d ** 2) - 4 * d + 2
    return 0


def get_cubic_matrix(start_i, start_j, d_i, d_j):
    return tuple((tuple((u(math.fabs(i + d_i)) * u(math.fabs(j + d_j)) for j in range(start_j, start_j + 4)))
                  for i in range(start_i, start_i + 4)))


print("tl")
cubic_tl_matrix = get_cubic_matrix(-2, -2, 0.25, 0.25)
print("tr")
cubic_tr_matrix = get_cubic_matrix(-2, -1, 0.25, -0.25)
print("bl")
cubic_bl_matrix = get_cubic_matrix(-1, -2, -0.25, 0.25)
print("br")
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

    # TODO: pad and shift indexes

    for new_i in range(new_rows):
        for new_j in range(new_cols):
            old_i, old_j = calc_coordinates(inverse_transformation, new_i, new_j)
            roi = get_roi_range(fract(old_i), fract(old_j))
            matrix = get_matrix(roi)
            start_i, start_j = roi

            try:
                matrix_roi = original_img[start_i: start_i + 4, start_j: start_j + 4]
                new_value = np.float32(matrix_roi).inner(matrix)
                new_img[new_i][new_j] = new_value
            except:  # TODO: Remove
                pass
