import numpy as np

class Detection(object):
    """Class represent a bounding box detection"""
    def __init__(self, tlwh, cofidence, feature) -> None:
        """
        :param tlwh: bounding box in format `(x, y, w, h)`
        :param feature: A feature vector that describes the object contained in this image.
        :param confidence (float): Detector confidence score.
        """
        self.confidence = float(cofidence)
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.feature = np.asarray(feature, dtype=np.float32)
        
    def to_tlbr(self):
        """
        Conver bounding box format to `(min x, min y, max x max y)`
        :return: The bounding box with `tlbr` format
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio, height)`
        where the aspect ratio is `width / height`.
        :return: The bounding box with `xyah` format
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    