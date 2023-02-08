from __future__ import absolute_import
from . import linear_assignment

import numpy as np


def iou(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Compute intersection over union
    :param bbox (ndarray): A bounding box
    :param candidates (ndarray):  A matrix of candidate bounding boxes per row

    :return: The intersection over union in [0, 1] between the `bbox` and each candidate. 
    A higher score means a larger fraction of the `bbox` is occluded by the candidate.
    :rtype: ndarray
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl, candidates_br = candidates[:,:2], candidates[:,:2] + candidates[:,2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks:list, detections:list, track_indices=None, detection_indices=None) -> np.ndarray:
    """
    An intersection over union distance metric.
    :param tracks:  A list of tracks.
    :param detections: A list of detections
    :param track_indices: A list of indices to tracks that should be matched
    :param detection_indices: A list of indices to detections that should be matched

    :return: a cost matrix of shape `len(track_indices)`, `len(detection_indices)` 
    where entry (i, j) is `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    :rtype: np.ndarray
    """

    if not track_indices:
        track_indices = np.arange(len(tracks))

    if not detection_indices:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    
    return cost_matrix
