from pathlib import Path
import numpy as np

from BraTS2020_semantic_segmentation.constants import CROP_XY, CROP_Z


def min_max_norm(img: np.ndarray) -> np.ndarray:
    """
    Function normalized image between 0-1

    Parameters
    ----------
    img
        image to normalized
    Returns
    -------
    Normalized image
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def crop_image(img: np.ndarray) -> np.ndarray:
    """
    Function cropping at least 3D array
    Parameters
    ----------
    img
        at least 3D image

    Returns
    -------
    Cropped image
    """
    return img[CROP_XY[0]:CROP_XY[1], CROP_XY[0]:CROP_XY[1], CROP_Z[0]:CROP_Z[1]]
