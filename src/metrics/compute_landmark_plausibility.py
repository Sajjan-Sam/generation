import numpy as np
def mahalanobis_score(features, mean, cov, eps=1e-6):
    mean = np.asarray(mean)
    cov = np.asarray(cov) + np.eye(len(mean)) * eps
    inv_cov = np.linalg.inv(cov)
    diffs = features - mean[None, :]
    d2 = np.einsum("bi,ij,bj->b", diffs, inv_cov, diffs)
    return d2
