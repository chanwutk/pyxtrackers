import numpy as np
cimport numpy as cnp

cnp.import_array()


def scale(cnp.ndarray[cnp.float64_t, ndim=2] detections,
          img_info, img_size):
    """Scale bounding boxes from model coordinates to original image coordinates.

    Computes ``scale = min(img_size[0]/img_info[0], img_size[1]/img_info[1])``
    and divides each bounding box (columns 0-3) by that factor.
    The score column (index 4) is left unchanged.

    Parameters
    ----------
    detections : ndarray, shape (N, 5), dtype float64
        Each row is ``[x1, y1, x2, y2, score]``.
    img_info : tuple[float, float]
        Original image dimensions ``(height, width)``.
    img_size : tuple[float, float]
        Model input dimensions ``(height, width)``.

    Returns
    -------
    ndarray, shape (N, 5), dtype float64
        A **copy** of *detections* with the bounding-box columns divided by
        the scale factor.
    """
    cdef:
        Py_ssize_t n = detections.shape[0]
        double img_h = <double>img_info[0]
        double img_w = <double>img_info[1]
        double s = min(<double>img_size[0] / img_h, <double>img_size[1] / img_w)
        Py_ssize_t i
        cnp.ndarray[cnp.float64_t, ndim=2] result

    if n == 0:
        return detections.copy()

    result = detections.copy()
    for i in range(n):
        result[i, 0] /= s
        result[i, 1] /= s
        result[i, 2] /= s
        result[i, 3] /= s

    return result
