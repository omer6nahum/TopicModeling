import numpy as np
from scipy.special import gamma
from numpy.linalg import norm

EPS = 1e-50


def volume(X):
    p = X.shape[1]
    sigma = np.std(X, axis=0)
    res = 2
    for j in range(p):
        res *= sigma[j]
    res *= (np.pi ** (p / 2))
    res /= p
    res /= gamma(p / 2)

    return res


def loglik_j(X_j, n):
    # X_j is the subset of points belongs to cluster j
    n_j, p = X_j.shape
    sigma_j = np.std(X_j, axis=0)
    return (n_j * np.log(n_j)) - \
           (n_j * np.log(n)) - \
           (0.5 * n_j * np.log(2 * np.pi)) - \
           (0.5 * n_j * np.sum(np.log(sigma_j**2 + EPS))) - \
           (0.5 * (n_j - 1))


def BICext(X, j1_indices, j2_indices):
    n_j1 = len(j1_indices)
    n_j2 = len(j2_indices)
    n = n_j1 + n_j2

    centroid_j1 = np.mean(X[j1_indices], axis=0)
    centroid_j2 = np.mean(X[j2_indices], axis=0)
    dists = np.array([(norm(X[i] - centroid_j1), norm(X[i] - centroid_j2)) for i in range(n)])
    dists = dists / np.expand_dims(np.sum(dists, axis=1), -1)
    assert dists.shape == (n, 2), dists.shape
    overlaps = dists[:, 0] * dists[:, 1]

    loglik_j1 = loglik_j(X[j1_indices], n)
    loglik_j2 = loglik_j(X[j2_indices], n)

    res = (-2 * (loglik_j1 + loglik_j2)) + \
          (4 * (np.log(n_j1) + np.log(n_j2))) + \
          (0.5 * n * np.sum(np.log(overlaps + EPS)))
    return res if res != -np.inf else np.nan

