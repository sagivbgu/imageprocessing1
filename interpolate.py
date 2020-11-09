import numpy as np
import math
from transform import calc_coordinates, does_exceed

MIN_INTENSITY = 0
MAX_INTENSITY = 255

# x, y locations relative to the examined pixel to start the cubic interpolation matrix from
# tl stands for top left, br stands for bottom right, etc.
TL_ROI = (-2, -2)
TR_ROI = (-1, -2)
BL_ROI = (-2, -1)
BR_ROI = (-1, -1)


def fract(num):
    return num - math.floor(num)


# Main interpolation loop, calling the desired interpolation function in each iteration
def interpolate(new_img, original_img, inverse_transformation, interpolation_func):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape

    padding = 2 if interpolation_func == interpolate_cubic else 0

    for new_y in range(padding, new_rows - padding):
        for new_x in range(padding, new_cols - padding):
            old_x, old_y = calc_coordinates(inverse_transformation, new_x, new_y)
            old_x += padding
            old_y += padding
            if does_exceed(round(old_x), round(old_y), old_rows, old_cols):
                continue
            new_value = interpolation_func(original_img, old_x, old_y, old_rows, old_cols)
            if new_value < MIN_INTENSITY:
                new_value = MIN_INTENSITY
            elif new_value > MAX_INTENSITY:
                new_value = MAX_INTENSITY
            new_img[new_y][new_x] = new_value


# Nearest neighbor interpolation
# Function signature follows interpolate function signature, though in this case the last arguments are not used
def interpolate_nearest(original_img, old_x, old_y, old_rows, old_cols):
    return original_img[round(old_y)][round(old_x)]


# Bilinear interpolation
def interpolate_bilinear(original_img, old_x, old_y, old_rows, old_cols):
    # Get boundaries, also handling the edges of the image
    top_y = 0 if round(old_y) == 0 else round(old_y) - 1
    bottom_y = top_y if top_y == old_rows - 1 else top_y + 1
    left_x = 0 if round(old_x) == 0 else round(old_x) - 1
    right_x = left_x if left_x == old_cols - 1 else left_x + 1

    width = math.fabs(old_x - left_x - 0.5)
    height = math.fabs(old_y - bottom_y - 0.5)

    intensity_top = (1 - width) * original_img[top_y][left_x] + width * original_img[top_y][right_x]
    intensity_bottom = (1 - width) * original_img[bottom_y][left_x] + width * original_img[bottom_y][right_x]
    return (1 - height) * intensity_bottom + height * intensity_top


# Bicubic interpolation
# Function signature follows interpolate function signature, though in this case the last arguments are not used
def interpolate_cubic(original_img, old_x, old_y, old_rows, old_cols):
    roi = get_roi(fract(old_x), fract(old_y))
    weight_matrix = get_weight_matrix(roi)
    start_x, start_y = roi
    old_x = math.floor(old_x)
    old_y = math.floor(old_y)
    matrix_roi = original_img[old_y + start_y: old_y + start_y + 4, old_x + start_x: old_x + start_x + 4]
    return calculate_cubic_new_value(matrix_roi, weight_matrix)


# Get locations relative to the examined pixel to start the cubic interpolation matrix from
def get_roi(fract_x, fract_y):
    if fract_y < 0.5 and fract_x < 0.5:
        return TL_ROI
    elif fract_y < 0.5 and fract_x >= 0.5:
        return TR_ROI
    elif fract_y >= 0.5 and fract_x < 0.5:
        return BL_ROI
    else:
        return BR_ROI


# The weight function in the interpolation formula. d is the distance
def u(d):
    if math.fabs(d) < 1:
        return 1.5 * (d ** 3) - 2.5 * (d ** 2) + 1
    elif math.fabs(d) < 2:
        return -0.5 * (d ** 3) + 2.5 * (d ** 2) - 4 * d + 2
    return 0


def get_cubic_matrix(start_x, start_y, d_x, d_y):
    return tuple((tuple((u(math.fabs(x + d_x)) * u(math.fabs(y + d_y)) for x in range(start_x, start_x + 4)))
                  for y in range(start_y, start_y + 4)))


cubic_tl_matrix = get_cubic_matrix(*TL_ROI, 0.25, 0.25)
cubic_tr_matrix = get_cubic_matrix(*TR_ROI, -0.25, 0.25)
cubic_bl_matrix = get_cubic_matrix(*BL_ROI, 0.25, -0.25)
cubic_br_matrix = get_cubic_matrix(*BR_ROI, -0.25, -0.25)


def get_weight_matrix(roi):
    if roi == TL_ROI:
        return cubic_tl_matrix
    if roi == TR_ROI:
        return cubic_tr_matrix
    if roi == BL_ROI:
        return cubic_bl_matrix
    if roi == BR_ROI:
        return cubic_br_matrix
    else:
        raise ValueError("Invalid roi")


def calculate_cubic_new_value(mat_roi, mat):
    mat_roi = np.float32(mat_roi).flatten()
    mat = np.float32(mat).flatten()
    return np.inner(mat_roi, mat)
