import numpy as np

class Detection(object):
    """Class represent a bounding box detection"""
    def __init__(self, tlwh:np.ndarray, confidence:np.ndarray, 
                 class_name:np.ndarray, feature:np.ndarray) -> None:
        """
        :param tlwh (np.ndarray): bounding box in format `(x, y, w, h)`
        :param confidence (np.ndarray): Detector confidence score.
        :param class_name (np.ndarray): Detector name
        :param feature (np.ndarray | NoneType): A feature vector 
        that describes the object contained in this image.
        """
        self.confidence = float(confidence)
        self.class_name = class_name
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
    
    def get_class(self) -> np.ndarray:
        """
        :return: the class name
        :rtype: np.ndarray
        """
        return self.class_name
    