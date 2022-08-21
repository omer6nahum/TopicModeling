from abc import ABC, abstractmethod
import numpy as np
from utils import volume


class Merge(ABC):
    @abstractmethod
    def fit_transform(self, X, y):
        pass


class NearestCentroidVolume(Merge):
    @staticmethod
    def criterion(X, y, c1, c2):
        # suppose we merge c1 and c2 into c
        # compare volume of c to volume of c1 and c2  (lower volume is better)

        p = X.shape[1]  # dimension
        # Volume comparison
        c = (y == c1) | (y == c2)
        org_clusters_volume = p * (volume(X[y == c1]) + volume(X[y == c2]))
        new_cluster_volume = volume(X[c])
        if new_cluster_volume < org_clusters_volume:
            return True
        return False

    def fit_transform(self, X, y):
        clusters = np.unique(y)
        clusters_centroids = {c: np.mean(X[y == c], axis=0) for c in clusters}
        merged_clusters = []
        new_assignments = np.array(y)
        merges = dict()

        for c, centroid in clusters_centroids.items():
            if c in merged_clusters:
                continue

            # this is not an optimal matching, but an order dependent
            # nearest cluster (based on distance from centroid) among other clusters that were not merged yet
            other_centroids = [(other_c, np.linalg.norm(other_cent - centroid))
                               for other_c, other_cent in clusters_centroids.items()
                               if other_c not in ([c] + merged_clusters)]
            if len(other_centroids) == 0:
                continue
            nearest = min(other_centroids, key=lambda x: x[1])

            if self.criterion(X, y, c, nearest[0]):
                # merge
                merged_clusters += [c, nearest[0]]
                new_id = np.max(new_assignments) + 1
                new_assignments[y == c] = new_id
                new_assignments[y == nearest[0]] = new_id
                merges[new_id] = (c, nearest[0])

        return new_assignments, merges
