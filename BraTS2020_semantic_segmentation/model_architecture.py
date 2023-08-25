import tensorflow as tf

kernel_init = 'he_uniform'


def simple_3d_unet_model(height, width, depth, n_channels, n_classes):
    inputs = tf.keras.Input((height, width, depth, n_channels))
    s = inputs

    # Contraction path
    c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same',
                                input_shape=(height, width, depth, n_channels), data_format='channels_last')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same', data_format='channels_last')(c1)

    c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c2)
    p2 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(c2)

    c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c3)
    p3 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(c3)

    c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c4)
    p4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(c4)

    c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c6)

    u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c7)

    u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c8)

    u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same', data_format='channels_last')(c9)

    outputs = tf.keras.layers.Conv3D(n_classes, (1, 1, 1), activation='softmax')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # model.summary()

    return model

# model = simple_unet_model(128, 128, 128, 3, 4)
# print(model.input_shape)
# print(model.output_shape)
