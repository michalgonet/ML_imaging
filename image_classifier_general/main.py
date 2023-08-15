import click
import constants
import tensorflow as tf

from image_classifier_general.utils_data import load_config, generate_datasets
from image_classifier_general.model_architecture import get_model


@click.command()
@click.option('--flag', type=click.Choice(['train', 'test', 'predict'], case_sensitive=False))
def main(flag: str):
    config = load_config(constants.CONFIG_PATH)
    train_ds, val_ds, test_ds, label_names = generate_datasets(config)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    if flag == 'train':
        model = get_model(config)
        model.fit(
            train_ds,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1,
            validation_data=val_ds
        )
        model.save(constants.MODELS_PATH / str(constants.MODEL_VERSION))

    if flag == 'test':
        pass
    if flag == 'predict':
        pass


if __name__ == '__main__':
    main()
