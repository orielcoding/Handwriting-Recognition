import argparse
import os
import cv2 as cv
import numpy as np


def padding(img, original_width, original_height, target_width, target_height):
    """
    pads an images with white background evenly on all sides
    :param img: the image that needs padding
    :param original_width: original width of the image
    :param original_height: original height of the image
    :param target_width: the target width of the new image
    :param target_height: the target height of the new image
    :return: a new padded image
    """
    padding_height_start = int((target_height - original_height) / 2)
    padding_height_end = padding_height_start + original_height

    padding_width_start = int((target_width - original_width) / 2)
    padding_width_end = padding_width_start + original_width

    padded_img = np.full([target_height, target_width, 1], 255) # creates an "empty" image (white pixels)
    padded_img[padding_height_start:padding_height_end, padding_width_start:padding_width_end, :] = img

    return padded_img


def resize(img, new_height, new_width):
    """
    calculates the new size of the based of the ratio between the new width/height and the original width/height this
    is done separately to avoid enlarging any dimension and pad it instead.
    The image is then resized and padded accordingly.
    :param img: the image that needs resizing
    :param new_height: the new height of the image
    :param new_width: the new width of the image
    :return: a resized padded image
    """
    original_width = img.shape[1]
    original_height = img.shape[0]
    scale_w = new_width / original_width
    scale_h = new_height / original_height

    if min(scale_w, scale_h) == scale_w:
        resize_width = new_width
        resize_height = int(resize_width * original_height / original_width)
    else:
        resize_height = new_height
        resize_width = int(resize_height * original_width / original_height)

    new_img = cv.resize(img, (resize_width, resize_height))
    return padding(new_img, resize_width, resize_height, new_width, new_height)


def prepare_img(img, img_w, img_h, bw):
    """
    resizing the original image to the desired dimensions
    :param img: the original img
    :param img_w: the desired width
    :param img_h: the desired height
    :param bw: binary image output? (black and white)
    :return: a new resized processed image
    """
    img = cv.imread(img)
    img = resize(img,img_h, img_w)
    img = np.uint8(img)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    if bw:
        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img = img.astype(np.float32)
    img /= 255
    return img


def prepare_and_save_image(filename, new_width, new_height, bw, destination):
    """
    calls the preprocessing function and saves the output
    """
    img = prepare_img(filename, new_width, new_height, bw)
    cv.imwrite(destination, img)


def process_batch(args):
    """
    process a batch of images (folder)
    """
    input_folder = args.input_folder
    dest_folder = args.dest_folder
    os.makedirs(dest_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(dest_folder, filename)

            prepare_and_save_image(input_file, args.new_width, args.new_height, args.BW, output_file)


def process_single_image(args):
    """
    processes a single image
    """
    input_file = args.input_file

    if not os.path.exists(input_file):
        print(f"input file '{input_file}' does not exist")
        return

    output_file = input_file.rsplit('.', 1)[0] + '_new.png'
    prepare_and_save_image(input_file, args.new_width, args.new_height, args.BW, output_file)


def parse():
    """
    args parser
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(prog='preprocessing',
                                     description='preprocess images (resize, color, noise filtering)')
    parser.add_argument('-batch', action='store_true')
    parser.add_argument('-input_folder', type=str, help='path to folder of images')
    parser.add_argument('-dest_folder', type=str, help='destination folder for resized images')
    parser.add_argument('-input_file', type=str, help='path to input file')
    parser.add_argument('-BW', action='store_true', help='transform the image to binary coloring (black and white)')
    parser.add_argument('-new_height', type=int, default=64, help='the new height after resize (recommended 64)')
    parser.add_argument('-new_width', type=int, default=512, help='the new width after resize (recommended 512)')

    return parser.parse_args()



def main():
    args = parse()

    if args.batch:
        process_batch(args)
    else:
        process_single_image(args)


if __name__ == "__main__":
    main()
