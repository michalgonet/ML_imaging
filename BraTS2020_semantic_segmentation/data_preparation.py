from pathlib import Path

import numpy as np
from keras.utils import to_categorical
import tensorflow as tf

from BraTS2020_semantic_segmentation import constants
from BraTS2020_semantic_segmentation.utils.data_utils import load_nifti, get_nifti_filepaths_sorted_list, \
    save_tfrecord, save_tfrecord2
from BraTS2020_semantic_segmentation.utils.math_utils import min_max_norm, crop_image


def prepare_train_data() -> None:
    """
    Function generates cropped images and labels from raw data into the Data/BraTS2020/Process/Train directory.
    Images are labels are store as .npy files. Images are 4D stock (H x W x slices x N) where N is image modality
    [T1ce, T2, Flair]

    """
    t1ce_list = get_nifti_filepaths_sorted_list(flag='train', pattern='t1ce')
    t2_list = get_nifti_filepaths_sorted_list(flag='train', pattern='t2')
    flair_list = get_nifti_filepaths_sorted_list(flag='train', pattern='flair')
    mask_list = get_nifti_filepaths_sorted_list(flag='train', pattern='seg')

    for t1ce_path, t2_path, flair_path, mask_path in zip(t1ce_list, t2_list, flair_list, mask_list):
        t1ce, t2, flair, mask = load_nifti(t1ce_path), load_nifti(t2_path), load_nifti(flair_path), load_nifti(
            mask_path)
        t1ce_n, t2_n, flair_n = min_max_norm(t1ce), min_max_norm(t2), min_max_norm(flair)
        mask[mask == 4] = 3

        stack_cropped = crop_image(np.stack([t1ce_n, t2_n, flair_n], axis=3))
        mask_cropped = to_categorical(crop_image(mask.astype(np.uint8)), num_classes=4)
        mask_cropped = mask_cropped.astype(np.uint8)
        # val, counts = np.unique(mask, return_counts=True)
        # if (1 - (counts[0] / counts.sum())) > 0.01:
        #     temp_mask = to_categorical(mask_cropped, num_classes=4)

        idx = mask_path.name[-11:-8]
        # print(Path(constants.PROCESS_TRAIN_IMG_PATH, f'Image_{idx}.npy'))
        np.save(str(Path(constants.PROCESS_TRAIN_IMG_PATH, f'image_{idx}.npy')), stack_cropped)
        np.save(str(Path(constants.PROCESS_TRAIN_LBL_PATH, f'label_{idx}.npy')), mask_cropped)
        # save_tfrecord2(
        #     data=stack_cropped,
        #     label=mask_cropped,
        #     path=constants.TRAIN_TFRECORD_PATH,
        #     file_name=f'image_{mask_path.name[-11:-8]}'
        # )


def prepare_test_data() -> None:
    """
    Function generates cropped images and labels from raw data into the Data/BraTS2020/Process/Test directory.
    Images are labels are store as .npy files. Images are 4D stock (H x W x slices x N) where N is image modality
    [T1ce, T2, Flair]

    """
    t1ce_list = get_nifti_filepaths_sorted_list(flag='test', pattern='t1ce')
    t2_list = get_nifti_filepaths_sorted_list(flag='test', pattern='t2')
    flair_list = get_nifti_filepaths_sorted_list(flag='test', pattern='flair')

    for t1ce_path, t2_path, flair_path in zip(t1ce_list, t2_list, flair_list):
        t1ce, t2, flair, = load_nifti(t1ce_path), load_nifti(t2_path), load_nifti(flair_path)
        t1ce_n, t2_n, flair_n = min_max_norm(t1ce), min_max_norm(t2), min_max_norm(flair)

        idx = t1ce_path.name[-12:-9]
        stack_cropped = crop_image(np.stack([t1ce_n, t2_n, flair_n], axis=3))
        np.save(str(Path(constants.PROCESS_TEST_IMG_PATH, f'image_{idx}.npy')), stack_cropped)

        # save_tfrecord(
        #     data=stack_cropped,
        #     path=constants.TEST_TFRECORD_PATH,
        #     file_name=f'image_{t1ce_path.name[-12:-9]}',
        #     flag='image')
