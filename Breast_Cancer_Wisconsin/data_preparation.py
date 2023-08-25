import pandas as pd

from Breast_Cancer_Wisconsin.constants import DATASET_PATH


def data_preparation():
    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={'diagnosis': 'Label'})
    df['Label'].value_counts()
    df_cleaned = df.dropna(axis=1)

    y = df_cleaned["Label"].values
    X = df_cleaned.drop(labels=["Label", "id"], axis=1)

    return X, y
