import numpy as np

def non_max_suppression(boxes:np.ndarray, max_bbox_overlap:float,
                        scores=None) -> list:
    """
    Suppress overlapping detections.
    :param boxes (np.ndarray): Array of ROIs `(x, y, width, height)`.
    :param max_bbox_overlap (float): ROIs that overlap more than this values are suppressed
    :param scores: Detector confidence score.
    :return: indices of detections that have survived non-maxima suppression.
    """
    if not boxes:
        return []
    
    boxes, pick = boxes.astype(np.float), []
    (x1, y1, x2, y2) = (boxes[:, 0], boxes[:, 1], 
                        boxes[:, 2] + boxes[:, 0], boxes[:, 3] + boxes[:, 1])
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores) if scores is not None else np.argsort(y2)

    while idxs:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
