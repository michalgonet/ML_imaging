from sklearn import metrics, svm
from sklearn.preprocessing import MinMaxScaler


def svm_classifier():
    return svm.SVC(kernel='linear', C=1, random_state=42)


def svm_training(x_train, y_train):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    model = svm_classifier()
    return model.fit(x_train, y_train)


def svm_test(model, x_test, y_test):
    scaler = MinMaxScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)
    y_pred = model.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)
