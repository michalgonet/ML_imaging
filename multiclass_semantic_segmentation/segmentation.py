import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf

from multiclass_semantic_segmentation.unet_structure import multi_unet_model
# from multiclass_semantic_segmentation.unet_structure_vgg16 import multi_unet_model
# from multiclass_semantic_segmentation.unet_structure_resnet import multi_unet_model
images_path = "C:\\Michal\\Programming\\Repositories_MG\\ML_imaging\\Data\\sandstone_data_for_ML\\full_labels_for_deep_learning\\128_patches\\images"
labels_path = "C:\\Michal\\Programming\\Repositories_MG\\ML_imaging\\Data\\sandstone_data_for_ML\\full_labels_for_deep_learning\\128_patches\\labels"

SIZE_X = 128
SIZE_Y = 128
n_classes = 4

train_images, train_masks = [], []

for img_path, mask_path in zip(glob.glob(images_path), glob.glob(labels_path)):
    for image, label in zip(glob.glob(os.path.join(img_path, "*.tif")), glob.glob(os.path.join(mask_path, "*.tif"))):
        img = cv2.imread(image, 0)
        lbl = cv2.imread(label, 0)
        train_images.append(img)
        train_masks.append(lbl)

train_images = np.array(train_images)
train_masks = np.array(train_masks)
# plt.subplot(1, 2, 1)
# plt.imshow(train_images[0])
# plt.subplot(1, 2, 2)
# plt.imshow(train_masks[0])
# plt.show()

label_encoder = preprocessing.LabelEncoder()
n, h, w = train_masks.shape
train_mask_1d = train_masks.reshape(-1, 1)
train_mask_1d_encoded = label_encoder.fit_transform(train_mask_1d)
train_masks_encoded_org_shape = train_mask_1d_encoded.reshape(n, h, w)

train_images = np.expand_dims(train_images, axis=3)
train_images = tf.keras.utils.normalize(train_images, axis=1)
train_mask_input = np.expand_dims(train_masks_encoded_org_shape, axis=3)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_mask_input, test_size=0.10, random_state=1)

train_masks_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_mask_cat = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_mask_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(train_mask_1d_encoded),
                                                  y=train_mask_1d_encoded)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


def get_model():
    return multi_unet_model(n_classes=n_classes, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channel=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_cat,
                    batch_size=32,
                    verbose=1,
                    epochs=100,
                    validation_data=(X_test, y_test_cat),
                    shuffle=False)

model.save('multi_segmentation_100_epochs_cat_cross_entropy.hdf5')
#
_, acc = model.evaluate(X_test, y_test_cat)
print(f'Accuracy is = {round(100 * acc, 2)} %')

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
