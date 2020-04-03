from data_land_marker import LandMarker

from image_classifier import Classifier
from camera_classifier import CameraClassifier

IMAGES_DIR = 'data/images/'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

INITIAL_CSV = 'data/ds_original.csv'
FINAL_CSV = 'data/ds_classes_equalized.csv'


def main():
    land_marker = LandMarker(landmark_predictor_path=PREDICTOR_PATH)
    rf_classifier = Classifier(csv_path=FINAL_CSV, algorithm='SVM', land_marker=land_marker)

    """
    from data_preparer import PreProcessor, DatasetBuilder
    
    # Pre-process data
    PreProcessor(data_dir=IMAGES_DIR).preprocess() 

    # Build dataset as csv
    DatasetBuilder(data_dir=IMAGES_DIR, class_feature='emotion', landmarker=landmarker).build(target=INITIAL_CSV)
    """

    CameraClassifier(classifier_model=rf_classifier).execute()


if __name__ == "__main__":
    main()
    print('success')
