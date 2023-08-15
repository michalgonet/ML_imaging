TRANSFER_LEARNING = {
    "resnet50": {
        "layer_mean_1chanel": "conv1_conv",
        "output_layer": "conv4_block6_out",
        "decode_layers_names": ["conv3_block4_out", "conv2_block3_out", "conv1_relu", "input_1"],
        "network_name": "Res50_U-Net"
    },
    "vgg16": {
        "layer_mean_1chanel": "block1_conv1",
        "output_layer": "block5_conv3",
        "decode_layers_names": ["block4_conv3", "block3_conv3", "block2_conv2", "block1_conv2"],
        "network_name": "VGG16_U-Net"
    }
}
TL_WEIGHTS: str = 'imagenet'
