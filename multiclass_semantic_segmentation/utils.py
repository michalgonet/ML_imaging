import os
import json
import glob
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn import preprocessing
from sklearn.utils import class_weight
from PIL import Image

from multiclass_semantic_segmentation import constants
from multiclass_semantic_segmentation.classes import Config
from multiclass_semantic_segmentation.unet_structure import multi_unet_model


def load_config(config_path: Path) -> Config:
    with open(str(config_path), 'r') as f:
        config_data = json.load(f)
    return Config(**config_data)


def load_data(config: Config) -> tuple[np.ndarray, np.ndarray]:
    train_images, train_masks = [], []

    for img_path, mask_path in zip(glob.glob(config.images_path), glob.glob(config.labels_path)):
        for image, label in zip(glob.glob(os.path.join(img_path, "*.tif")),
                                glob.glob(os.path.join(mask_path, "*.tif"))):
            img = cv2.imread(image, 0)
            lbl = cv2.imread(label, 0)
            train_images.append(img)
            train_masks.append(lbl)

    train_images = np.array(train_images)
    train_images = np.expand_dims(train_images, axis=3)
    train_images = tf.keras.utils.normalize(train_images, axis=1)

    train_masks = np.array(train_masks)
    return train_images, train_masks


def encode_mask_pixel_values(train_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_encoder = preprocessing.LabelEncoder()
    n, h, w = train_masks.shape
    train_mask_1d = train_masks.reshape(-1, 1)
    train_mask_1d_encoded = label_encoder.fit_transform(train_mask_1d)
    train_masks_encoded_org_shape = train_mask_1d_encoded.reshape(n, h, w)
    train_mask_input = np.expand_dims(train_masks_encoded_org_shape, axis=3)

    return train_mask_input, train_mask_1d_encoded


def convert_to_categorical(config: Config, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks_cat = tf.keras.utils.to_categorical(data, num_classes=config.n_classes)
    data_cat = masks_cat.reshape((data.shape[0], data.shape[1], data.shape[2], config.n_classes))
    return data_cat, masks_cat


def _avg_weights(weights: np.ndarray) -> np.ndarray:
    average_weights = np.mean(weights, axis=-2)
    average_weights = average_weights[:, :, np.newaxis, :]
    return average_weights


def get_pixel_weights(train_mask_1d: np.ndarray) -> np.ndarray:
    return class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(train_mask_1d),
                                             y=train_mask_1d)


def _get_pretrained_model(architecture: str, input_sz: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    architecture_models = {
        'resnet50': (tf.keras.applications.ResNet50, 'resnet50'),
        'vgg16': (tf.keras.applications.VGG16, 'vgg16')
    }

    if architecture not in architecture_models:
        raise ValueError(f'Architecture {architecture} unknown')

    model_class, info_key = architecture_models[architecture]
    base_model = model_class(include_top=False, weights=constants.TL_WEIGHTS)
    info = constants.TRANSFER_LEARNING[info_key]

    base_model_config = base_model.get_config()
    base_model_config["layers"][0]["config"]["batch_input_shape"] = (None, input_sz[0], input_sz[1], input_sz[2])

    updated_model = tf.keras.Model.from_config(base_model_config)
    updated_model_config = updated_model.get_config()
    updated_model_layers_name = [updated_model_config["layers"][x]["name"] for x in
                                 range(len(updated_model_config['layers']))]

    for layer in base_model.layers:
        if layer.name in updated_model_layers_name:
            if layer.get_weights():
                target_layer = updated_model.get_layer(layer.name)

                if layer.name == info["layer_mean_1chanel"]:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    weights_single_channel = _avg_weights(weights)
                    target_layer.set_weights([weights_single_channel, biases])
                    target_layer.trainable = False
                else:
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = False

    model_1ch = tf.keras.Model.from_config(updated_model.get_config())

    for layer in model_1ch.layers:
        layer.trainable = False

    model_output = model_1ch.get_layer(info["output_layer"]).output

    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(model_output)
    conv6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same')(up1)
    merge6 = tf.keras.layers.concatenate([model_1ch.get_layer(info["decode_layers_names"][0]).output, conv6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(up2)
    merge7 = tf.keras.layers.concatenate([model_1ch.get_layer(info["decode_layers_names"][1]).output, conv7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    conv8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge8 = tf.keras.layers.concatenate([model_1ch.get_layer(info["decode_layers_names"][2]).output, conv8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    conv9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge9 = tf.keras.layers.concatenate([model_1ch.get_layer(info["decode_layers_names"][3]).output, conv9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=model_1ch.input, outputs=outputs, name=info["network_name"])

    return model


def get_model(config: Config, input_shape) -> tf.keras.Model:
    if config.transfer_learning:
        return _get_pretrained_model(
            architecture=config.transfer_network,
            input_sz=input_shape,
            num_classes=config.n_classes
        )
    else:
        return multi_unet_model(
            n_classes=config.n_classes,
            input_sz=input_shape,
        )


def _load_img(filepath_list):
    images = []
    for i, image_path in enumerate(filepath_list):
        image = np.array(Image.open(image_path))
        images.append(image)

    return np.array(images)


def imageLoader(img_list, mask_list, batch_size):
    all_files = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < all_files:
            limit = min(batch_end, all_files)
            x = _load_img(img_list[batch_start:limit])
            y = _load_img(mask_list[batch_start:limit])
            yield (np.expand_dims(x, axis=3), np.expand_dims(y, axis=3))
            batch_start += batch_size
            batch_end += batch_size
