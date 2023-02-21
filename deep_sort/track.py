import numpy as np
from . import kalman_filter, detection

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track(object):
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """
    def __init__(self, mean:np.ndarray, covariance:np.ndarray,
                 track_id:int, n_init:int, max_age:int, 
                 feature:np.ndarray=None, class_name=None) -> None:
        """
        :param mean (np.ndarray): Mean vector of the initial state distribution.
        :param covariance (np.ndarray): Covariance matrix of the initial state distribution.
        :param track_id (int): A unique track identifier.
        :param n_init (int): Number of consecutive detections before the track is confirmed.
        :param max_age (int): The maximum number of consecutive misses before 
        the track state is set to `Deleted`.
        :param feature (np.ndarray): Feature vector of the detection this track originates from.
        """
        self.mean, self.covariance = mean, covariance
        self.hits, self.age, self.track_id = 1, 1, track_id
        self._n_init, self._max_age, self.class_name = n_init, max_age, class_name
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

    def to_tlwh(self) -> np.ndarray:
        """
        Get current position in bounding box format 
        `(top left x, top left y, width, height)`.
        :return: The bounding box.
        :rtype: np.ndarray
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_tlbr(self) -> np.ndarray:
        """
        Get current position in bounding box format 
        `(min x, miny, max x, max y)`.
        :return: The bounding box.
        :rtype: np.ndarray
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    

    def predict(self, kf:kalman_filter.KalmanFilter) -> None:
        """
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        :param kf (kalman_filter.KalmanFilter): The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf:kalman_filter.KalmanFilter, 
               detection:detection.Detection) -> None:
        """
        Perform Kalman filter measurement update step and update the feature cache.
        :param kf (kalman_filter.KalmanFilter): The Kalman filter.
        :param detection (detection.Detection): The associated detection.
        """
        self.mean, self.covariance = kf.update(self.mean, 
                                               self.covariance, 
                                               detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """ Mark this track as missed (no association at the current time step). """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> None:
        """ Returns True if this track is tentative (unconfirmed). """
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> None:
        """ Returns True if this track is confirmed. """
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> None:
        """ Returns True if this track is dead and should be deleted. """
        return self.state == TrackState.Deleted
    
    def get_class(self):
        return self.class_name
