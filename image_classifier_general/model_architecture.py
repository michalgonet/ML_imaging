import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from classes import Config
from utils_data import define_preprocessing, define_augmentation


def get_model(config: Config):
    input_shape = (config.batch_size, config.image_size, config.image_size, config.n_channels)

    model = keras.models.Sequential([
        define_preprocessing(config),
        define_augmentation(config),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(config.n_classes, activation='softmax')
        # keras.layers.Dense(N_CLASSES, activation='sigmoid')

    ])

    model.build(input_shape=input_shape)

    if config.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        raise ValueError(f'Invalid optimizer: {config.optimizer}')

    if config.loss_function == 'sparse_categorical_crossentropy':
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        raise ValueError(f'Invalid loss function: {config.loss_function}')

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=config.metrics
    )

    return model
