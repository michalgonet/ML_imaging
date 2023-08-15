from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, \
    Activation, Concatenate
import numpy as np
from keras import Sequential
from keras.applications import VGG16


def avg_weights(weights):
    average_weights = np.mean(weights, axis=-2)
    average_weights = average_weights[:, :, np.newaxis, :]
    return average_weights


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def multi_unet_model(n_classes, img_height, img_width, img_channel):
    base_model = VGG16(include_top=False, weights='imagenet')
    base_model_config = base_model.get_config()
    base_model_config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, img_channel)

    updated_model = Model.from_config(base_model_config)
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

    vgg_1ch = updated_model
    s1 = vgg_1ch.get_layer("block1_conv2").output
    s2 = vgg_1ch.get_layer("block2_conv2").output
    s3 = vgg_1ch.get_layer("block3_conv3").output
    s4 = vgg_1ch.get_layer("block4_conv3").output
    b1 = vgg_1ch.get_layer("block5_conv3").output
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)

    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(d2)

    model = Model(inputs, outputs, name="VGG16_U-Net")

    return new_model
