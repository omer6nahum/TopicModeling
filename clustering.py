from sklearn.cluster import KMeans
from abc import ABC, abstractmethod



class Clustering(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def get_labels(self):
        pass


class KMeansClustering(Clustering):
    def __init__(self, k):
        self.k = k
        self.model = KMeans(n_clusters=k, random_state=2)

    def fit(self, X):
        self.model.fit(X)

    def get_labels(self):
        return self.model.labels_
