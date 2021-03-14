import glob
from random import shuffle
import h5py
import numpy as np
import cv2
import math
import csv
import time
import matplotlib.pyplot as plt


def normalize_and_write_data_into_h5_file(dest_path, filepaths, n_px, n_channels=3):
    """
        This function converts images to numpy arrays then writes the array data into a h5 file.
        dest_path - the name of the file with full path that is being created
        filepaths - source image file paths which is being converted to numpy arrays,
        dataset_name - name of the dataset
        n_px - number of pixels - will be used as image's height and width
        n_channels - 3 for rgb
    """

    data_shape = (len(filepaths), n_px * n_px * n_channels)
    dataset_name = "birds"

    with h5py.File(dest_path, 'a') as f:
        f.create_dataset(dataset_name, data_shape, np.float32)
        print(filepaths)
        for i in range(len(filepaths)):
            filepath = filepaths[i]
            img = cv2.imread(filepath)
            img = cv2.resize(img, (n_px, n_px),
                             interpolation=cv2.INTER_AREA)  # pick inter_area since we are shrinking the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize the image - convert the each pixel value between 0 and 1
            img = img / 255
            # Reshape the image - roll it up into a column vector
            img = img.ravel()

            # img[None] makes it a proper array instead of rank 1 array
            f[dataset_name][i, ...] = img[None]


def write_labels_into_h5_file(dest_path, labels):
    dataset_name = "labels"

    with h5py.File(dest_path, 'a') as f:
        f.create_dataset(dataset_name, (len(labels),), np.int16)
        f[dataset_name][...] = labels


def write_file_name_into_h5_file(dest_path, names):
    dataset_name = "file_name"
    with h5py.File(dest_path, 'a') as f:
        names = [n.encode("ascii", "ignore") for n in names]
        f.create_dataset(dataset_name, (len(names),), data=names)


def convert_images_to_data_in_h5_file(src_img_filepath_pattern, dest_h5_file_path, label_dict, n_px,
                                      n_channels=3, batch_size=1024):
    """
    :param src_img_filepath_pattern: a pattern that stores image data that will be stored in h5
    :param dest_h5_file_path: the destination of the h5 data
    :param n_px: size of the image after resizing
    :param label_dict: label information, file name: label
    :param n_channels:
    :param batch_size:
    :return:
    """
    # Returns a list of filepaths matching the pattern given as parameter
    src_filepaths = glob.glob(src_img_filepath_pattern)
    file_names = [s.split("\\")[-1] for s in src_filepaths]
    # Create Labels based upon the substring contained in the filename
    labels = [label_dict[name] for name in file_names]

    # The zip(source_filepaths, labels) combines each element of source_filepaths list
    # with each element of labels list forming a pair (tuple). t is the list which contains these tuples
    t = list(zip(src_filepaths, labels, file_names))

    # Shuffle the list
    shuffle(t)

    # Get the shuffled filepaths & labels
    src_filepaths, labels, file_names = zip(*t)

    # Number of images
    m = len(src_filepaths)
    n_complete_batches = math.ceil(m / batch_size)

    for i in range(n_complete_batches):
        print('Creating file', (i + 1))

        dest_file_path = dest_h5_file_path + str(i + 1) + ".h5"

        start_pos = i * batch_size
        end_pos = min(start_pos + batch_size, m)
        src_filepaths_batch = src_filepaths[start_pos: end_pos]
        labels_batch = labels[start_pos: end_pos]
        file_name_batch = file_names[start_pos: end_pos]

        normalize_and_write_data_into_h5_file(dest_file_path, src_filepaths_batch, n_px)
        write_labels_into_h5_file(dest_file_path, labels_batch)
        write_file_name_into_h5_file(dest_file_path, file_name_batch)


if __name__ == '__main__':
    with open('D:/Course/CSE_455/Final_Project/birds21wi/labels.csv', mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            label_dict = {rows[0]: rows[1] for rows in reader}
    del label_dict["path"]

    src_filepath_pattern = 'D:/Course/CSE_455/Final_Project/birds21wi/train/*/*.jpg'
    dest_filepath = 'D:/Course/CSE_455/Final_Project/h5files/'
    n_px = 128
    n_channels = 3

    tic = time.process_time()
    convert_images_to_data_in_h5_file(src_filepath_pattern, dest_filepath, label_dict, n_px, n_channels)
    toc = time.process_time()
    print('Time taken for creating the h5 file is', (toc - tic) * 1000, 'ms')

    # test image
    destination_filepath = 'D:/Course/CSE_455/Final_Project/h5files/1.h5'
    with h5py.File(destination_filepath, "r") as f:
        print(list(f.keys()))

        x = f["birds"][:]
        y = f["labels"][:]
        z = f["file_name"][:]

        print('x shape =', x.shape, '| y shape =', y.shape, '| z shape =', z.shape)
