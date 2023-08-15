import random
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.metrics import MeanIoU

images_path = "C:\\Michal\\Programming\\Repositories_MG\\repo-for-practice\\Data\\sandstone_data_for_ML\\full_labels_for_deep_learning\\128_patches\\images"
labels_path = "C:\\Michal\\Programming\\Repositories_MG\\repo-for-practice\\Data\\sandstone_data_for_ML\\full_labels_for_deep_learning\\128_patches\\labels"

model = tf.keras.models.load_model('models/multi_segmentation_100_epochs_cat_cross_entropy.hdf5')

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

_, acc = model.evaluate(X_test, y_test_cat)
print(f'Accuracy is = {round(100 * acc, 2)} %')

y_predict = model.predict(X_test)
y_predict_argmax = np.argmax(y_predict, axis=3)
n_classes = 4
IOU_keras = tf.keras.metrics.MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_predict_argmax)
print(f'MeanIOU = {IOU_keras.result().numpy()}')

# IOU for each class
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
class1_IoU = values[0, 0] / (
        values[0, 0] + values[0, 1] + values[0, 2] + values[0, 3] + values[1, 0] + values[2, 0] + values[3, 0])
class2_IoU = values[1, 1] / (
        values[1, 1] + values[1, 0] + values[1, 2] + values[1, 3] + values[0, 1] + values[2, 1] + values[3, 1])
class3_IoU = values[2, 2] / (
        values[2, 2] + values[2, 0] + values[2, 1] + values[2, 3] + values[0, 2] + values[1, 2] + values[3, 2])
class4_IoU = values[3, 3] / (
        values[3, 3] + values[3, 0] + values[3, 1] + values[3, 2] + values[0, 3] + values[1, 3] + values[2, 3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0][:, :, None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

#####################################################################

# Predict on large image

# Apply a trained model on large image

# from patchify import patchify, unpatchify
#
# large_image = cv2.imread('large_images/large_image.tif', 0)
# # This will split the image into small images of shape [3,3]
# patches = patchify(large_image, (128, 128), step=128)  # Step=256 for 256 patches means no overlap
#
# predicted_patches = []
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         print(i, j)
#
#         single_patch = patches[i, j, :, :]
#         single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
#         single_patch_input = np.expand_dims(single_patch_norm, 0)
#         single_patch_prediction = (model.predict(single_patch_input))
#         single_patch_predicted_img = np.argmax(single_patch_prediction, axis=3)[0, :, :]
#
#         predicted_patches.append(single_patch_predicted_img)
#
# predicted_patches = np.array(predicted_patches)
#
# predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128, 128))
#
# reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
# plt.imshow(reconstructed_image, cmap='gray')
# # plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')
#
# plt.hist(reconstructed_image.flatten())  # Threshold everything above 0
#
# # final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# # # plt.imshow(final_prediction)
# #
# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.title('Large Image')
# plt.imshow(large_image, cmap='gray')
# plt.subplot(222)
# plt.title('Prediction of large Image')
# plt.imshow(reconstructed_image, cmap='jet')
# plt.show()
