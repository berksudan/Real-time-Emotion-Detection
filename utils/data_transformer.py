import math
from typing import List, Tuple

import numpy as np


def rectangular_to_bounding_box(rect):
    # take a bounding predicted by dlib and convert it to the format (x, y, w, h) as usual in OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


def shape_to_np(shape, size: int):
    return np.asarray([(shape.part(i).x, shape.part(i).y) for i in range(0, size)])


def points_to_vectors(x_list: Tuple, y_list: Tuple, exclude_vector_base:bool) -> List:
    landmarks_vectorised = []

    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    x_central = [(x - x_mean) for x in x_list]
    y_central = [(y - y_mean) for y in y_list]

    for x, y, w, z in zip(x_central, y_central, x_list, y_list):
        if not exclude_vector_base:
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
        mean_np = np.asarray((y_mean, x_mean))
        coordinates_np = np.asarray((z, w))
        dist = np.linalg.norm(coordinates_np - mean_np)
        landmarks_vectorised.append(dist) # Distance
        landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi)) # Alpha
    return landmarks_vectorised
