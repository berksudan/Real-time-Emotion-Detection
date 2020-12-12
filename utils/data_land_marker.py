from typing import Tuple, List, Optional

import cv2
import dlib
import numpy as np

from utils import data_transformer as dt

LAND_MARK_POINTS_SIZE = 68
CLAHE_CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


class LandMarker:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

    def __init__(self, landmark_predictor_path: str):
        self.predictor = dlib.shape_predictor(landmark_predictor_path)  # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()  # initialize dlib's face detector

    def img_to_landmarks(self, img, exclude_vector_base: bool = False) -> List[Tuple]:
        detections = self.detector(img, 1)
        if len(detections) == 0:
            return []
        face_land_marks_list = []
        for k, d in enumerate(detections):  # For all detected face instances individually
            shape = self.predictor(img, d)  # Draw Facial Landmarks with the predictor class
            x_list = tuple(float(shape.part(i).x) for i in range(LAND_MARK_POINTS_SIZE))
            y_list = tuple(float(shape.part(i).y) for i in range(LAND_MARK_POINTS_SIZE))
            face_land_marks_list.append(tuple(dt.points_to_vectors(x_list, y_list, exclude_vector_base)))
        return face_land_marks_list

    def img_path_to_landmarks(self, img_path: str, gray_scale: bool = True, exclude_vector_base: bool = True):
        if gray_scale:
            return self.img_to_landmarks(dlib.load_grayscale_image(img_path), exclude_vector_base)
        return self.img_to_landmarks(dlib.load_rgb_image(img_path), exclude_vector_base)

    def img_to_landmark_points(self, img: np.ndarray) -> List:
        detections = self.detector(img, 1)
        if len(detections) < 1:
            return [None]
        landmark_points_list = []
        for (i, rect) in enumerate(detections):
            shape = self.predictor(img, rect)
            landmark_points_list.append(dt.shape_to_np(shape, size=LAND_MARK_POINTS_SIZE))
        return landmark_points_list

    def img_to_rectangles(self, img: np.ndarray) -> List[Tuple]:
        detections = self.detector(img, 1)
        if len(detections) == 0:
            return [(0, 0, 0, 0)]

        rectangles = []
        for (i, rect) in enumerate(detections):  # loop over the face detections
            # convert dlib's rectangle to a OpenCV-style bounding box, i.e. (x, y, w, h)
            (x, y, w, h) = dt.rectangular_to_bounding_box(rect)

            rectangles.append((x, y, w, h))  # Assumed that number of face in picture = 1
        return rectangles
