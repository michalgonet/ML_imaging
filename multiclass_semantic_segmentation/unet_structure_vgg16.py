import numpy as np
import tensorflow as tf


def avg_weights(weights):
    average_weights = np.mean(weights, axis=-2)
    average_weights = average_weights[:, :, np.newaxis, :]
    return average_weights


def multi_unet_model(n_classes, img_height, img_width, img_channel):
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

    base_model_config = base_model.get_config()
    base_model_config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, img_channel)

    updated_model = tf.keras.Model.from_config(base_model_config)
    updated_model_config = updated_model.get_config()
    updated_model_layers_name = [updated_model_config["layers"][x]["name"] for x in
                                 range(len(updated_model_config['layers']))]

    first_conv_name = updated_model_layers_name[1]
    for layer in base_model.layers:
        if layer.name in updated_model_layers_name:
            if layer.get_weights():
                target_layer = updated_model.get_layer(layer.name)

                if layer.name in first_conv_name:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    weights_single_channel = avg_weights(weights)
                    target_layer.set_weights([weights_single_channel, biases])
                    target_layer.trainable = False
                else:
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = False

    vgg_1ch = tf.keras.Model.from_config(updated_model.get_config())

    for layer in vgg_1ch.layers:
        layer.trainable = False

    vgg_1ch_output = vgg_1ch.get_layer('block5_conv3').output

    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(vgg_1ch_output)
    conv6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same')(up1)
    merge6 = tf.keras.layers.concatenate([vgg_1ch.get_layer('block4_conv3').output, conv6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(up2)
    merge7 = tf.keras.layers.concatenate([vgg_1ch.get_layer('block3_conv3').output, conv7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    conv8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge8 = tf.keras.layers.concatenate([vgg_1ch.get_layer('block2_conv2').output, conv8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    conv9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge9 = tf.keras.layers.concatenate([vgg_1ch.get_layer('block1_conv2').output, conv9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=vgg_1ch.input, outputs=outputs, name="VGG16_U-Net")

    return model
