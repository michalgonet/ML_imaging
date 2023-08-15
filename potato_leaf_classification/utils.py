import numpy as np
import tensorflow as tf
import numpy as np


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, shuffle=True, shuffle_size=10000):
    ds_length = len(ds)

    train_length = int(train_split * ds_length)
    val_length = int(val_split * ds_length)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_ds = ds.take(train_length)
    val_ds = ds.skip(train_length).take(val_length)
    test_ds = ds.skip(train_length).skip(val_length)

    return train_ds, val_ds, test_ds


def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    return predicted_class, confidence
