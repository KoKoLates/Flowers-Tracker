import numpy as np


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns the closest distance 
    to any sample that has been observed so far.
    """
    def __init__(self, metric:str, matching_threshold:float, budget:int=None)-> None:
        """
        :param metric (str): Either "euclidean" or "cosine".
        :param matching_threshold (float): The matching threshold. 
        Samples with larger distance are considered an invalid match.
        :param budget (int): If not None, fix samples per class to at most this number. 
        Removes the oldest samples when the budget is reached.
        """
        if metric in ("euclidean", "cosine"):
            self._metric = (NearestNeighborDistanceMetric._nn_euclidean_distance if metric == "euclidean" 
                            else NearestNeighborDistanceMetric._nn_euclidean_distance)
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        
        self.samples = {}
        self.budget = budget
        self.matching_threshold = matching_threshold

    def partial_fit(self, features:np.ndarray, targets:np.ndarray, 
                    active_targets) -> None:
        """
        Update the distance metric with new data.
        :param feature (np.ndarray): An `N x M` matrix of N features of dimensionality M.
        :param targets (np.ndarray): An integer array of associated target identities.
        :param active_targets: A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features:np.ndarray, targets) -> np.ndarray:
        """
        Compute distance between features and targets.
        :param features (np.ndarray): An `N x M` matrix of N features of dimensionality M.
        :param targets: A list of targets to match the given `features` against.
        :return: a cost matrix of shape len(targets), len(features), where element (i, j) 
        contains the closest squared distance between `targets[i]` and `features[j]`.
        :rtype: np.ndarray
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
    
    @staticmethod
    def _pdist(a, b) -> np.ndarray:
        """
        Compute pair-wise squared distance between points in `a` and `b`.
        :param a: An `N x M` matrix of N samples of dimensionality M.
        :param b: An `L x M` matrix of L samples of dimensionality M.

        :return: a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
        :rtype: np.ndarray
        """
        a, b = np.asarray(a), np.asarray(b)
        if not len(a) or not len(b):
            return np.zeros((len(a), len(b)))
        a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
        r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))
        return r2

    @staticmethod
    def _cosine_distance(a, b, data_is_normalized:bool=False) -> np.ndarray:
        """
        Compute pair-wise cosine distance between points in `a` and `b`.
        """
        if not data_is_normalized:
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        return 1. - np.dot(a, b.T)

    @staticmethod
    def _nn_euclidean_distance(x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Helper function for nearest neighbor distance metric (Euclidean).
        :param x (np.ndarray): A matrix of N row-vectors (sample points).
        :param y (np.ndarray): A matrix of M row-vectors (query points).
        :return: A vector of length M that contains for each entry in `y` 
        the smallest Euclidean distance to a sample in `x`.
        :rtype: np.ndarray
        """
        distances = NearestNeighborDistanceMetric._pdist(x, y)
        return np.maximum(0.0, distances.min(axis=0))

    @staticmethod
    def _nn_cosine_distance(x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Helper function for nearest neighbor distance metric (cosine).
        :param x (np.ndarray): A matrix of N row-vectors (sample points).
        :param y (np.ndarray): A matrix of y row-vectors (sample points).
        :return: A vector of length M that contains for each entry in `y` 
        the smallest cosine distance to a sample in `x`.
        :rtype: np.ndarray
        """
        distances = NearestNeighborDistanceMetric._cosine_distance(x, y)
        return distances.min(axis=0)
    