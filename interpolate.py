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

# def interpolation_cubic(new_img, original_img, inverse_transformation):
#     new_rows, new_cols = new_img.shape
#     old_rows, old_cols = original_img.shape
#     for new_i in range(new_rows):
#         for new_j in range(new_cols):
#             old_i, old_j = calc_coordinates(inverse_transformation, new_i, new_j)
#             top_i = 0 if round(old_i) == 0 else round(old_i) - 1
#             bottom_i = old_rows - 1 if round(old_i) == old_rows - 1 else round(old_i)
#             left_j = 0 if round(old_j) == 0 else round(old_j) - 1
#             right_j = old_cols if round(old_j) == old_cols - 1 else round(old_j)
#             width = math.fabs(0.5 - fract(old_i))
#             height = math.fabs(0.5 - fract(old_j))
#
#             intensity_top = (1 - width) * original_img[top_i][left_j] + width * original_img[top_i][right_j]
#             intensity_bottom = (1 - width) * original_img[bottom_i][left_j] + width * original_img[bottom_i][right_j]
#             new_img[new_i][new_j] = (1 - height) * intensity_bottom + height * intensity_top
