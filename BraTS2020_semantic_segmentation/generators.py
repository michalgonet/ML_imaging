import numpy as np


def load_img(images_path_list):
    images = []
    for i, image_name in enumerate(images_path_list):
        image = np.load(image_name)
        images.append(image)

    return np.array(images)


def image_loader(img_list, mask_list, batch_size):
    all_files = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < all_files:
            limit = min(batch_end, all_files)
            x = load_img(img_list[batch_start:limit])
            y = load_img(mask_list[batch_start:limit])
            yield x, y
            batch_start += batch_size
            batch_end += batch_size
