from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import numpy as np
from utils import BICext, volume


class Split(ABC):
    @abstractmethod
    def fit_transform(self, X, y):
        pass


class KMeansVolume(Split):
    def __init__(self):
        self.model = KMeans(n_clusters=2)
        self.eps = 1e-30

    def fit_transform(self, X, y):
        clusters = np.unique(y)
        new_assignments = np.array(y)
        splits = dict()
        bic_dict = dict()
        p = X.shape[1]  # dimension

        for cluster in clusters:
            cur_points = X[y == cluster]
            if len(cur_points) <= 5:
                continue
            cur_indices = np.where(y == cluster)[0]

            # try to find 2 clusters within this cluster
            self.model.fit(cur_points)
            cur_2_clusters = self.model.labels_
            # calc bic
            bic_dict[cluster] = BICext(cur_points, np.where(cur_2_clusters == 0)[0], np.where(cur_2_clusters == 1)[0])
            if np.mean(cur_2_clusters) > 0.95 or np.mean(cur_2_clusters) < 0.05:
                # this means that induces clusters are highly imbalanced
                continue

            # Volume comparison
            if volume(cur_points) > self.eps + p * (volume(cur_points[cur_2_clusters == 0]) +
                                                    volume(cur_points[cur_2_clusters == 1])):
                # split
                new_id0 = np.max(new_assignments) + 1
                new_id1 = np.max(new_assignments) + 2
                new_assignments[cur_indices[cur_2_clusters == 0]] = new_id0
                new_assignments[cur_indices[cur_2_clusters == 1]] = new_id1
                splits[cluster] = (new_id0, new_id1)

        # "fix" by bic - another criterion
        mean_bic = np.nanmean(list(bic_dict.values()))
        std_bic = np.nanstd(list(bic_dict.values()))
        bic_dict = {k: ((v-mean_bic)/std_bic) for k, v in bic_dict.items()}

        print(f'{len(splits)} optional splits (by volume criterion), only ', end='')
        for cluster in clusters:
            if np.sum(y == cluster) <= 5:
                continue
            if (not np.isnan(bic_dict[cluster])) and (np.abs(bic_dict[cluster]) < 1):
                # cancel split, since it is not different from other splits
                try:
                    i1, i2 = splits[cluster]
                except KeyError:
                    continue
                new_assignments[new_assignments == i1] = cluster
                new_assignments[new_assignments == i2] = cluster
                del splits[cluster]
        print(f'{splits} splits performed.')
        return new_assignments, splits
