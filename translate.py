from sys import argv
import cv2
import transform
from interpolate import interpolate, interpolate_nearest, interpolate_bilinear, interpolate_cubic
from transform import create_empty_img


def load_image_file(path):
    try:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise Exception("Error reading image file {0}".format(path)) from e


def load_trans_file(path):
    try:
        with open(path, 'r') as img_file:
            return (tuple(line.split()) for line in img_file.readlines())
    except Exception as e:
        raise Exception("Error reading transformation file {0}".format(path)) from e


def translate_image(image_path, transformations_path, quality):
    img = load_image_file(image_path)
    transformations = load_trans_file(transformations_path)

    new_img, mat, inv_mat = transform.apply_trans_on_img(transformations, img)
    print("Inv mat:")
    print(inv_mat)

    if quality == "N":
        interpolate(new_img, img, inv_mat, interpolate_nearest)
    elif quality == "B":
        interpolate(new_img, img, inv_mat, interpolate_bilinear)
    elif quality == "C":
        img = add_margins(img)
        interpolate(new_img, img, inv_mat, interpolate_cubic)
    else:
        print("Invalid Input")
        return

    cv2.imwrite('out_{0}.png'.format(quality), new_img)


def add_margins(img, add_h=2, add_w=2):
    """
    Pad an image with a given padding sizes
    :param img: The image to pad
    :param add_h: The number of pixels to add as a padding to each side (height)
    :param add_w: The number of pixels to add as a padding to each side (width)
    :return: The padded image
    """
    h, w = img.shape
    new_h = h + add_h * 2
    new_w = w + add_w * 2

    new_image = create_empty_img(new_h, new_w)
    for y in range(h):
        for x in range(w):
            new_x = x + add_w
            new_y = y + add_h
            new_image[new_y][new_x] = img[y][x]

    # Fill the first and last rows with the same values as the first and last rows of the original image
    for y in range(add_h):
        for x in range(w):
            new_image[y][x + add_w] = img[0][x]
            new_image[-1 - y][x + add_w] = img[-1][x]

    # Fill the first and columns of each row with the same values as the first and last columns of the original image
    for x in range(add_w):
        for y in range(h):
            new_image[y + add_h][x] = img[y][0]
            new_image[y + add_h][-1 - x] = img[y][-1]

    return new_image


def main():
    if len(argv) != 4:
        print("Usage {0} image_file tran_file quality".format(argv[0]))
        return

    translate_image(argv[1], argv[2], argv[3])


if __name__ == '__main__':
    main()
