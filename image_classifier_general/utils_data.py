import json
import keras
import tensorflow as tf

from classes import Config
import constants


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return Config(**config_data)


def generate_datasets(config: Config) -> tuple[keras.datasets, keras.datasets, keras.datasets, list[str]]:
    dataset = keras.preprocessing.image_dataset_from_directory(
        directory=constants.DATASET_PATH,
        shuffle=config.shuffle,
        image_size=(config.image_size, config.image_size),
        batch_size=config.batch_size
    )
    label_names = dataset.class_names

    dataset_length = len(dataset)
    train_length = int(config.train_split * dataset_length)
    validation_length = int(config.val_split * dataset_length)

    if config.shuffle:
        dataset = dataset.shuffle(config.shuffle_size, config.seed)

    train_ds = dataset.take(train_length)
    val_ds = dataset.skip(train_length).take(validation_length)
    test_ds = dataset.skip(train_length).skip(validation_length)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names


def define_preprocessing(config: Config):
    processes = []
    if config.resizing:
        processes.append(keras.layers.experimental.preprocessing.Resizing(config.image_size, config.image_size))
    if config.rescaling:
        processes.append(keras.layers.experimental.preprocessing.Rescaling(1.0 / 255))

    return keras.Sequential(processes)


def define_augmentation(config: Config):
    processes = []
    if config.random_flip:
        processes.append(keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        if config.random_rotation:
            processes.append(keras.layers.experimental.preprocessing.RandomRotation(config.random_rotation))

    return keras.Sequential(processes)
