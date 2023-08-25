from pathlib import Path

from sklearn.model_selection import train_test_split

from multiclass_semantic_segmentation.utils import load_config, load_data, encode_mask_pixel_values, \
    convert_to_categorical, get_model, get_pixel_weights

config = load_config(Path("../multiclass_semantic_segmentation/configurations/config.json"))

# Data preparation
train_images, train_masks = load_data(config)
train_mask_input, train_mask_1d_encoded = encode_mask_pixel_values(train_masks)
X_train, X_test, y_train, y_test = train_test_split(train_images, train_mask_input, test_size=config.test_size)
y_train_cat, masks_train_cat = convert_to_categorical(config, y_train)
y_test_cat, masks_test_cat = convert_to_categorical(config, y_test)
pixel_weights = get_pixel_weights(train_mask_1d_encoded)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# Model definition
# input_shape = (128, 128, 1)
model = get_model(config, input_shape)
model.compile(
    optimizer=config.optimizer,
    loss=config.loss_function,
    metrics=config.metrics)

# Training
img_path = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/images'
mask_path = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/labels'
batch_size = 4
img_list = list(Path(img_path).rglob('*.tif'))
mask_path = list(Path(mask_path).rglob('*.tif'))

steps_per_epoch = len(img_list) // batch_size

from utils import imageLoader

img_generator = imageLoader(img_list, mask_path, batch_size)

# history = model.fit(X_train, y_train_cat,
#                     batch_size=config.batch_size,
#                     verbose=1,
#                     epochs=config.epochs,
#                     validation_data=(X_test, y_test_cat),
#                     shuffle=config.shuffle)

history = model.fit(
    img_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    verbose=1,
    # validation_data=train_gen,
    # validation_steps=steps_per_epoch,
)
# Save Model
model.save(
    f'models/N_classes_{config.n_classes}_'
    f'Epochs_{config.epochs}_'
    f'Optimizer_{config.optimizer}_'
    f'Loss_{config.loss_function}_'
    f'Transfer_learning_{config.transfer_learning}_'
    f'Pretrained_network_{config.transfer_network}.hdf5')

_, acc = model.evaluate(X_test, y_test_cat)
print(f'Accuracy is = {round(100 * acc, 2)} %')
