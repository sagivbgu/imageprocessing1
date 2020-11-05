from sys import argv
import cv2
import transform
import interpolate


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

    # new_img, mat, inv_mat = transform.apply_trans_on_img(transformations, img)
    return transform.apply_trans_on_img(transformations, img)


def main():
    if len(argv) != 4:
        print("Usage {0} image_file tran_file quality".format(argv[0]))
        return

    translate_image(argv[1], argv[2], argv[3])


if __name__ == '__main__':
    main()
