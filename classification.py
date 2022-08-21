from sklearn.svm import SVC
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SVMClassifier(Classifier):
    def __init__(self):
        self.model = SVC(random_state=2, class_weight='balanced')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
