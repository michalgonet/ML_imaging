import click

from BraTS2020_semantic_segmentation.data_preparation import prepare_test_data, prepare_train_data
from BraTS2020_semantic_segmentation.training import train


@click.command()
@click.option('--flag', type=click.Choice(['prepare', 'train', 'test', 'predict'], case_sensitive=False))
def main(flag):
    if flag == 'prepare':
        pass
        # prepare_test_data()
        # prepare_train_data()
        # from BraTS2020_semantic_segmentation.utils.data_utils import load_tfrecords
        # file_paths ="C:\\Michal\\Programming\\Repositories_MG\\ML_imaging\\Data\\BraTS2020\\Process\\Train\\TFrecords\\image_003.tfrecord"
        # loaded_data = load_tfrecords(file_paths)

    elif flag == 'train':
        train()
    elif flag == 'test':
        raise NotImplementedError
    elif flag == 'predict':
        raise NotImplementedError
    else:
        raise ValueError(f'Flag {flag} unknown')


if __name__ == '__main__':
    main()
