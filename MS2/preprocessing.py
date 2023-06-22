import argparse
import os
import cv2 as cv
import numpy as np


def padding(img, original_width, original_height, target_width, target_height):
    """

    :param img:
    :param original_width:
    :param original_height:
    :param target_width:
    :param target_height:
    :return:
    """
    padding_height_start = int((target_height - original_height) / 2)
    padding_height_end = padding_height_start + original_height

    padding_width_start = int((target_width - original_width) / 2)
    padding_width_end = padding_width_start + original_width

    padded_img = np.full([target_height, target_width, 1], 255)
    padded_img[padding_height_start:padding_height_end, padding_width_start:padding_width_end, :] = img

    return padded_img


def resize(img, new_height, new_width):
    """

    :param img:
    :param new_height:
    :param new_width:
    :return:
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

    :param img:
    :param img_w:
    :param img_h:
    :param bw:
    :return:
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


def parse():
    """

    :return:
    """
    parser = argparse.ArgumentParser(prog='preprocessing',
                                     description='preprocess images (resize, color, noise filtering)')
    parser.add_argument('-batch', action='store_true')
    parser.add_argument('-batch_size', default=None, help='batch size (number of images)')
    parser.add_argument('-input_folder', type=str, help='path to folder of images')
    parser.add_argument('-dest_folder', type=str, help='destination folder for resized images')
    parser.add_argument('-input_file', type=str, help='path to input file')
    parser.add_argument('-BW', action='store_true', help='transform the image to binary coloring (black and white)')
    parser.add_argument('-new_height', type=int, default=64, help='the new height after resize (recommended 64)')
    parser.add_argument('-new_width', type=int, default=512, help='the new width after resize (recommended 512)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.batch:
        try:
            if not os.path.exists(args.dest_folder):
                os.makedirs(args.dest_folder)
        except OSError as err:
            print("destination folder was not entered or the path does not exist")

        if not args.batch_size:
            args.batch_size = len(os.listdir(args.input_folder))
        batch = 0

        for filename in os.listdir(args.input_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if batch == args.batch_size:
                    break

                img = prepare_img(os.path.join(args.input_folder, filename), args.new_width, args.new_height, args.BW)
                cv.imwrite(os.path.join(args.dest_folder, filename), img)
                batch += 1
    else:
        try:
            os.path.exists(args.input_file)
        except OSError as err:
            print("input file was not entered or it does not exist")
        img = prepare_img(args.input_file, args.new_width, args.new_height, args.BW)
        cv.imwrite(args.input_file.split('.')[0] + '_new.png', img)
