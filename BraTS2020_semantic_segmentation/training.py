from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

from BraTS2020_semantic_segmentation import constants
from BraTS2020_semantic_segmentation.generators import image_loader
from BraTS2020_semantic_segmentation.helpers import show_imagegen_example
# from BraTS2020_semantic_segmentation.model_architecture import simple_3d_unet_model
from BraTS2020_semantic_segmentation.model_architecture import simple_3d_unet_model

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

def train():
    train_img_list = list(Path(constants.PROCESS_TRAIN_IMG_PATH).glob('*.npy'))
    train_mask_list = list(Path(constants.PROCESS_TRAIN_LBL_PATH).glob('*.npy'))

    train_gen = image_loader(train_img_list, train_mask_list, constants.BATCH_SIZE)

    # img, mask = train_gen.__next__()
    # show_imagegen_example(img, mask)

    steps_per_epoch = len(train_img_list) // constants.BATCH_SIZE
    LR = 0.0001
    model = simple_3d_unet_model(height=128, width=128, depth=128, n_channels=1, n_classes=4)
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(LR),
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
    import segmentation_models_3D as sm

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

    LR = 0.0001
    optim = keras.optimizers.Adam(LR)

    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #               metrics=["accuracy"])

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=constants.EPOCHS,
        verbose=constants.VERBOSE,
        validation_data=train_gen,
        validation_steps=steps_per_epoch,
    )

    model.save(f'{constants.SAVED_MODELS}/ver_{constants.MODEL_VERSION}')
