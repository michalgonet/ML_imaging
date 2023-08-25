from pathlib import Path

import numpy as np
import nibabel as nib
import tensorflow as tf

from BraTS2020_semantic_segmentation import constants


def load_nifti(filepath: Path) -> np.ndarray:
    return nib.load(filepath).get_fdata()


def get_nifti_filepaths_sorted_list(flag: str, pattern: str) -> list[Path]:
    if flag == 'train':
        path = constants.TRAIN_PATH
    elif flag == 'test':
        path = constants.VALID_PATH
    else:
        raise ValueError(f'Flag {flag} unknown')

    return sorted(list(Path(path).rglob(f'*{pattern}.nii')))


def _serialize_data(data, flag):
    feature = {
        'data': None,
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=data.shape))
    }
    if flag == 'image':
        feature['data'] = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
    elif flag == 'label':
        feature['data'] = tf.train.Feature(int64_list=tf.train.Int64List(value=data.flatten()))
    else:
        raise ValueError(f'Invalid flag: {flag}')

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _serialize_data2(data, label):
    feature = {
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.flatten())),
        'shape_data': tf.train.Feature(int64_list=tf.train.Int64List(value=data.shape)),
        'shape_label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.shape))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def save_tfrecord(data: np.ndarray, path: str, file_name: str, flag: str) -> None:
    tfrecord_file_path = Path(path, f'{file_name}.tfrecord')
    with tf.io.TFRecordWriter(str(tfrecord_file_path)) as writer:
        example = _serialize_data(data, flag)
        writer.write(example)


def save_tfrecord2(data: np.ndarray, label: np.ndarray, path: str, file_name: str) -> None:
    tfrecord_file_path = Path(path, f'{file_name}.tfrecord')
    with tf.io.TFRecordWriter(str(tfrecord_file_path)) as writer:
        example = _serialize_data2(data, label)
        writer.write(example)


def parse_tfrecord(serialized_example):
    feature_description = {
        'data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'shape_data': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
        'shape_label': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64)
    }
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode features
    data = tf.io.decode_raw(parsed_example['data'], tf.float32)
    label = tf.io.decode_raw(parsed_example['label'], tf.int64)

    # Reshape data and label arrays
    data_shape = parsed_example['shape_data']
    label_shape = parsed_example['shape_label']
    data = tf.reshape(data, data_shape)
    label = tf.reshape(label, label_shape)

    return data, label


def load_tfrecords(file_paths):
    dataset = tf.data.TFRecordDataset(file_paths)
    parsed_dataset = dataset.map(parse_tfrecord)

    data_list = list(parsed_dataset.as_numpy_iterator())
    return data_list
# Example usage

