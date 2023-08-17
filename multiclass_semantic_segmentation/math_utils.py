from keras import backend as K
import numpy as np


def jacard_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Function returns IoU value for two ground truth and predicted images.
    It is defined as ratio of the Area of Overlap and Area of Union

    Parameters
    ----------
    y_true
        ground true image
    y_pred
        predicted image
    Returns
    -------
    IoU also known as Jaccard Index/Jaccard Similarity Coefficient/Intersection over Union
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Loss function using IoU. Negative sign is here because loss function is trying to minimize this value while the
    Jacard value is increasing during learning process.

    Parameters
    ----------
    y_true
        ground true image
    y_pred
        predicted image
    Returns
    -------
    Function returns negative value of Jaccard_coef
    """
    return -jacard_coef(y_true, y_pred)
