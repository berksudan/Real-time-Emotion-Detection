from utils.data_land_marker import LandMarker
from data_preparer import PreProcessor, DatasetBuilder

from image_classifier import Classifier
from camera_classifier import CameraClassifier
from os.path import isfile

DATASET_IMAGES_DIR = 'data/raw'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

DATASET_CSV = 'data/csv/dataset.csv'


def main():
    land_marker = LandMarker(landmark_predictor_path=PREDICTOR_PATH)

    if not isfile(DATASET_CSV):
        print('[INFO]', f'Dataset file: "{DATASET_CSV}" could not found.')
        # Pre-process data
        labeled_images = PreProcessor(dataset_parent_dir=DATASET_IMAGES_DIR).preprocess()

        # Build dataset as csv
        DatasetBuilder(labeled_images, class_col='emotion', land_marker=land_marker).build(target=DATASET_CSV)
    else:
        print('[INFO]', f'Dataset file: "{DATASET_CSV}" found.')

    rf_classifier = Classifier(csv_path=DATASET_CSV, algorithm='SVM', land_marker=land_marker)
    CameraClassifier(classifier_model=rf_classifier).execute()


if __name__ == "__main__":
    main()
    print('success')
