# import the necessary packages


import cv2.cv2 as cv2
import dlib
import numpy as np

import data_transformer as dt

LAND_MARK_POINTS_SIZE = 68
CLAHE_CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


class LandMarker:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

    def __init__(self, landmark_predictor_path: str):
        self.predictor = dlib.shape_predictor(landmark_predictor_path)  # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()  # initialize dlib's face detector

    def img_to_landmarks(self, img):
        detections = self.detector(img, 1)
        if len(detections) == 0:
            return None
        for k, d in enumerate(detections):  # For all detected face instances individually
            shape = self.predictor(img, d)  # Draw Facial Landmarks with the predictor class
            x_list = tuple(float(shape.part(i).x) for i in range(1, LAND_MARK_POINTS_SIZE))
            y_list = tuple(float(shape.part(i).y) for i in range(1, LAND_MARK_POINTS_SIZE))

            return tuple(dt.points_to_vectors(x_list, y_list))

    def img_to_landmark_points(self, img: np.ndarray):
        rectangles = self.detector(img, 1)
        for (i, rect) in enumerate(rectangles):
            shape = self.predictor(img, rect)
            return dt.shape_to_np(shape, size=LAND_MARK_POINTS_SIZE)


    def img_to_rectangle(self, img: np.ndarray):
        detections = self.detector(img, 1)
        if len(detections) == 0:
            return 0, 0, 0, 0

        for (i, rect) in enumerate(detections):  # loop over the face detections
            # convert dlib's rectangle to a OpenCV-style bounding box, i.e. (x, y, w, h)
            (x, y, w, h) = dt.rectangular_to_bounding_box(rect)

            return x, y, w, h  # Assumed that number of face in picture = 1
