from pathlib import Path
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from BraTS2020_semantic_segmentation.constants import TRAIN_PATH
from BraTS2020_semantic_segmentation.utils.math_utils import min_max_norm


def show_test_data():
    sample_flair = nib.load(Path(TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_flair.nii')).get_fdata()
    sample_flair_norm = min_max_norm(sample_flair)

    sample_t1 = nib.load(Path(TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1.nii')).get_fdata()
    sample_t1_norm = min_max_norm(sample_t1)

    sample_t1ce = nib.load(Path(TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1ce.nii')).get_fdata()
    sample_t1ce_norm = min_max_norm(sample_t1ce)

    sample_t2 = nib.load(Path(TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t2.nii')).get_fdata()
    sample_t2_norm = min_max_norm(sample_t2)

    sample_mask = nib.load(Path(TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_seg.nii')).get_fdata()
    sample_mask = sample_mask.astype(np.uint8)
    sample_mask[sample_mask == 4] = 3

    random_slice = random.randint(0, sample_mask.shape[2])

    plt.subplot(231)
    plt.imshow(sample_t1_norm[:, :, random_slice], cmap=plt.cm.gray)
    plt.title(f'T1 slice: {random_slice}')

    plt.subplot(232)
    plt.imshow(sample_t1ce_norm[:, :, random_slice], cmap=plt.cm.gray)
    plt.title(f'T1ce slice: {random_slice}')

    plt.subplot(233)
    plt.imshow(sample_t2_norm[:, :, random_slice], cmap=plt.cm.gray)
    plt.title(f'T2 slice: {random_slice}')

    plt.subplot(234)
    plt.imshow(sample_flair_norm[:, :, random_slice], cmap=plt.cm.gray)
    plt.title(f'Flair slice: {random_slice}')

    plt.subplot(235)
    plt.imshow(sample_mask[:, :, random_slice], cmap=plt.cm.jet)
    plt.title(f'Mask slice: {random_slice}')

    plt.show()


def show_imagegen_example(img, mask):
    img_num = random.randint(0, img.shape[0] - 1)
    test_img = img[img_num]
    test_mask = mask[img_num]
    test_mask = np.argmax(test_mask, axis=3)
    n_slice = random.randint(0, test_img.shape[2])
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
    plt.title('Image flair')

    plt.subplot(222)
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.title('Image t1ce')

    plt.subplot(223)
    plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
    plt.title('Image t2')

    plt.subplot(224)
    plt.imshow(test_mask[:, :, n_slice], cmap='jet')
    plt.title('Mask')

    plt.show()