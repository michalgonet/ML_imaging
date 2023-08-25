TRAIN_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_TrainingData'
VALID_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Raw/BraTS2020_ValidationData'

PROCESS_TRAIN_IMG_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Process/Train/images'
PROCESS_TRAIN_LBL_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Process/Train/labels'
PROCESS_TEST_IMG_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Process/Test/images'

TRAIN_TFRECORD_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Process/Train/TFrecords'
TEST_TFRECORD_PATH: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/Data/BraTS2020/Process/Test/TFrecords'

SAVED_MODELS: str = 'C:/Michal/Programming/Repositories_MG/ML_imaging/BraTS2020_semantic_segmentation/models'

CROP_XY: list = [56, 184]
CROP_Z: list = [13, 141]

BATCH_SIZE: int = 3
WEIGHTS: list[float] = [0.25, 0.25, 0.25, 0.25]
EPOCHS: int = 100
VERBOSE: int = 1
MODEL_VERSION: float = 1.0
