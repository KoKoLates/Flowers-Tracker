import numpy as np
import scipy.linalg as scl

from typing import Tuple

class KalmanFilter(object):
    """
    """

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

        # Table for the 0.95 quantile of the chi-square distribution with N degrees of
        # freedom (contains values for N = 1, 2, 3 ... 9).
        self.chi_square = {
            1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
            6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
        }

    def initialize(self, measurement: np.ndarray) -> Tuple(np.ndarray):
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

            






