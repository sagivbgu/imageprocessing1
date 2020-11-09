import numpy as np
import math
from transform import calc_coordinates, does_exceed

MIN_INTENSITY = 0
MAX_INTENSITY = 255

def interpolation_nearest(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape

    for new_y in range(new_rows):
        for new_x in range(new_cols):
            old_x, old_y = calc_coordinates(inverse_transformation, new_x, new_y)
            old_y = round(old_y)
            old_x = round(old_x)
            if not (does_exceed(old_x, old_y, old_rows, old_cols)):
                new_img[new_y][new_x] = original_img[old_y][old_x]


def fract(num):
    return num - math.floor(num)


# TODO: Exceed
def interpolation_bilinear(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape
    for new_y in range(new_rows):
        for new_x in range(new_cols):
            old_x, old_y = calc_coordinates(inverse_transformation, new_x, new_y)

            if does_exceed(old_x, old_y, old_rows, old_cols):  # TODO: Test this
                continue

            top_y = 0 if round(old_y) == 0 else round(old_y) - 1
            bottom_y = top_y if top_y == old_rows - 1 else top_y + 1
            left_x = 0 if round(old_x) == 0 else round(old_x) - 1
            right_x = left_x if left_x == old_cols - 1 else left_x + 1
            width = math.fabs(old_x - left_x - 0.5)
            height = math.fabs(old_y - bottom_y - 0.5)

            intensity_top = (1 - width) * original_img[top_y][left_x] + width * original_img[top_y][right_x]
            intensity_bottom = (1 - width) * original_img[bottom_y][left_x] + width * original_img[bottom_y][right_x]
            new_img[new_y][new_x] = (1 - height) * intensity_bottom + height * intensity_top


def u(d):
    if math.fabs(d) < 1:
        return 1.5 * (d ** 3) - 2.5 * (d ** 2) + 1
    elif math.fabs(d) < 2:
        return -0.5 * (d ** 3) + 2.5 * (d ** 2) - 4 * d + 2
    return 0


# TODO: Remove prints and tuple()s
def get_cubic_matrix(start_x, start_y, d_x, d_y):
    return tuple((tuple((u(math.fabs(x + d_x)) * u(math.fabs(y + d_y)) for x in range(start_x, start_x + 4)))
                  for y in range(start_y, start_y + 4)))


cubic_tl_matrix = get_cubic_matrix(-2, -2, 0.25, 0.25)
cubic_tr_matrix = get_cubic_matrix(-1, -2, -0.25, 0.25)
cubic_bl_matrix = get_cubic_matrix(-2, -1, 0.25, -0.25)
cubic_br_matrix = get_cubic_matrix(-1, -1, -0.25, -0.25)


def get_matrix(roi):
    if roi == (-2, -2):
        return cubic_tl_matrix
    if roi == (-1, -2):
        return cubic_tr_matrix
    if roi == (-2, -1):
        return cubic_bl_matrix
    if roi == (-1, -1):
        return cubic_br_matrix


def get_roi_range(fract_x, fract_y):
    if fract_y < 0.5 and fract_x < 0.5:
        return -2, -2  # tl
    elif fract_y < 0.5 and fract_x >= 0.5:
        return -1, -2  # tr
    elif fract_y >= 0.5 and fract_x < 0.5:
        return -2, -1  # bl
    else:
        return -1, -1  # br


def interpolation_cubic(new_img, original_img, inverse_transformation):
    new_rows, new_cols = new_img.shape
    old_rows, old_cols = original_img.shape

    # TODO: pad and shift indexes

    for new_y in range(new_rows):
        for new_x in range(new_cols):
            try:
                old_x, old_y = calc_coordinates(inverse_transformation, new_x, new_y)

                # if the pixel exceeds from the original pic,
                # we don't need to interpolate
                if does_exceed(old_x, old_y, old_rows, old_cols):
                    continue

                roi = get_roi_range(fract(old_x), fract(old_y))
                matrix = get_matrix(roi)
                start_x, start_y = roi
                old_x = math.floor(old_x)
                old_y = math.floor(old_y)

                matrix_roi = original_img[old_y + start_y: old_y + start_y + 4, old_x + start_x: old_x + start_x + 4]
                new_value = calculate_cubic_new_value(matrix_roi, matrix)

                if (new_value < 0 or new_value > 255):
                    print("new value: {0}, x: {1}, y: {2}".format(new_value, new_x, new_y))

                if new_value < MIN_INTENSITY:
                    new_value = MIN_INTENSITY
                elif new_value > MAX_INTENSITY:
                    new_value = MAX_INTENSITY
                new_img[new_y][new_x] = new_value
            except ValueError:  # Index exceeds matrix
                pass
                # TODO: Delete
                # if (old_x > 10 and old_x < 250 and old_y > 10 and old_y < 250):
                #     print("x: {0}, y: {1}".format(old_x, old_y))


def calculate_cubic_new_value(mat_roi, mat):
    mat_roi = np.float32(mat_roi).flatten()
    mat = np.float32(mat).flatten()
    return np.inner(mat_roi, mat)
