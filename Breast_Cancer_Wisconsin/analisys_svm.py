import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from Breast_Cancer_Wisconsin.data_preparation import data_preparation
from Breast_Cancer_Wisconsin.training import svm_training, svm_test, svm_classifier
from Breast_Cancer_Wisconsin.constants import MODE, K_FOLDS, SEED, SHUFFLE, TEST_SIZE

mode = MODE

X, y = data_preparation()
if mode == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    model = svm_training(X_train, y_train)
    accuracy = svm_test(model, X_test, y_test)

elif mode == 2:
    from sklearn.pipeline import Pipeline

    X_array = X.to_numpy()
    model = svm_classifier()
    steps = [('scaler', MinMaxScaler()), ('model', model)]
    pipline = Pipeline(steps=steps)
    cv = KFold(n_splits=K_FOLDS, random_state=SEED, shuffle=SHUFFLE)
    scores = cross_val_score(pipline, X_array, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # for score in scores:
    #     print(f'Accuracy: {score}')
    print(f'Mean acc = {np.mean(scores)}')
