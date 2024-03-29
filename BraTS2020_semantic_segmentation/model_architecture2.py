import tensorflow as tf


def simple_3d_unet_model(height, width, depth, n_channels, n_classes):
    inputs = tf.keras.Input((height, width, depth, n_channels))

    conv1 = tf.keras.layers.Conv3D(8, 3, activation='relu', padding='same', data_format="channels_last")(inputs)
    conv1 = tf.keras.layers.Conv3D(8, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv3D(64, 2, activation='relu', padding='same')(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv3D(32, 2, activation='relu', padding='same')(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv3D(16, 2, activation='relu', padding='same')(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=-1)
    conv8 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv3D(8, 2, activation='relu', padding='same')(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=-1)
    conv9 = tf.keras.layers.Conv3D(8, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv3D(8, 3, activation='relu', padding='same')(conv9)
    # conv10 = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(conv9)
    # model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    outputs = tf.keras.layers.Conv3D(n_classes, (1, 1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
