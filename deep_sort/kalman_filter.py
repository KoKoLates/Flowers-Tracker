import numpy as np
import scipy.linalg as scl

class KalmanFilter(object):
    """
    """
    # Table for the 0.95 quantile of the chi-square distribution with N degrees of
    # freedom (contains values for N = 1, 2, 3 ... 9).
    chi_square = {
        1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
        6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
    }

    def __init__(self) -> None:
        ndim, dt = 4, 1

        # create Kalman Filter model matrics
        self._update_mat = np.eye(ndim, ndim * 2)
        self._motion_mat = np.eye(ndim * 2, ndim * 2)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initialize(self, measurement: np.ndarray):
        """
        Create track from unassociated measurement.
        :param measurement (ndarray): Bounding box coordinates `(x, y, a, h)` 
        with center position `(x, y)`, aspect ratio `a`, and height `h`. 

        :return: the mean vector and covariance matrix of the new track
        :rtype: (ndarray, ndarray)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return (mean, covariance)

    def project(self, mean:np.ndarray, covariance:np.ndarray):
        """
        Project state distribution to measurement space.
        :param mean (ndarray): The state's mean vector
        :param covariance (ndarray): The state's covariance matrix

        :return: the projected mean and covariance matrix of the given state estimate.
        :rtype: (ndarray, ndarray)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return (mean, covariance + innovation_cov)

    def update(self, mean:np.ndarray, covariance:np.ndarray, measurement:np.ndarray):
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scl.cho_factor(projected_cov, lower=True, check_finite=False)
        innovation = measurement - projected_mean
        kalman_gain = scl.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean:np.ndarray, covariance:np.ndarray, 
                        measurements:np.ndarray, only_position:bool=False) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scl.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
