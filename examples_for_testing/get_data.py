# import numpy as np
# import nibabel as nib
# import glob
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt
# from tifffile import imsave
#
# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler()
# ##########################
# # This part of the code to get an initial understanding of the dataset.
# #################################
# # PART 1: Load sample images and visualize
# # Includes, dividing each image by its max to scale them to [0,1]
# # Converting mask from float to uint8
# # Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
# # Visualize
# ###########################################
# # View a few images
#
# # Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
#
# TRAIN_DATASET_PATH = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData'
# # VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
#
#
# # # # images lists harley
# # t1_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
# t2_list = sorted(glob.glob('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData/*/*t2.nii'))
# t1ce_list = sorted(glob.glob('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData/*/*t1ce.nii'))
# flair_list = sorted(glob.glob('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData/*/*flair.nii'))
# mask_list = sorted(glob.glob('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData/*/*seg.nii'))
#
# # Each volume generates 18 64x64x64x4 sub-volumes.
# # Total 369 volumes = 6642 sub volumes
#
# for img in range(len(t2_list)):  # Using t1_list as all lists are of same size
#     print("Now preparing image and masks number: ", img)
#
#     temp_image_t2 = nib.load(t2_list[img]).get_fdata()
#     temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
#         temp_image_t2.shape)
#
#     temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
#     temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
#         temp_image_t1ce.shape)
#
#     temp_image_flair = nib.load(flair_list[img]).get_fdata()
#     temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
#         temp_image_flair.shape)
#
#     temp_mask = nib.load(mask_list[img]).get_fdata()
#     temp_mask = temp_mask.astype(np.uint8)
#     temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
#     # print(np.unique(temp_mask))
#
#     temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
#
#     # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
#     # cropping x, y, and z
#     temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
#     temp_mask = temp_mask[56:184, 56:184, 13:141]
#
#     val, counts = np.unique(temp_mask, return_counts=True)
#
#     if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
#         print("Save Me")
#         temp_mask = to_categorical(temp_mask, num_classes=4)
#         np.save('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/example/images/image_' + str(img) + '.npy', temp_combined_images)
#         np.save('C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/example/masks/mask_' + str(img) + '.npy', temp_mask)
#
#     else:
#         print("I am useless")
#
#     ################################################################
# # Repeat the same from above for validation data folder OR
# # Split training data into train and validation
#
# """
# Code for splitting folder into train, test, and val.
# Once the new folders are created rename them and arrange in the format below to be used
# for semantic segmentation using data generators.
#
# pip install split-folders
# """
import splitfolders  # or import split_folders

input_folder = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/example/'
output_folder = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/example2/input_data_128/'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)  # default values
########################################