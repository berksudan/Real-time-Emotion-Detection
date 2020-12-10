import warnings

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def preprocess_data(data):
    # split data into input and target variable(s)
    x = data.drop("emotion", axis=1)
    y = data["emotion"]

    # standardize the dataset
    x_scaled = StandardScaler().fit_transform(x)

    # split into train and test set
    return train_test_split(x_scaled, y, stratify=y, test_size=0.40, random_state=42)


def evaluate_model(x_train, y_train, x_test, y_test, classifier):
    # Calculate Model Accuracy using Support Vector Machines
    classifier.fit(x_train, y_train)
    y_hat = classifier.predict(x_test)
    print(classification_report(y_test, y_hat))
    print("Total F1 Accuracy Score:", accuracy_score(y_test, y_hat))
    print('-' * 48)


def preview_data(data, label):
    print(f'[INFO] Emotion Counts:\n{data[label].value_counts()}')
    print('[INFO] Data Columns:', list(data.columns))
    print('[INFO] Data Shape:', data.shape)


def run_emotion_detection_evaluator(
        dataset_csv: str,
        predictor_path: str = None,
        dataset_images_dir: str = None):
    from os.path import isfile

    if not isfile(dataset_csv):  # If data-set not built before.
        from utils.data_land_marker import LandMarker
        land_marker = LandMarker(landmark_predictor_path=predictor_path)
        print('[INFO]', f'Dataset file: "{dataset_csv}" could not found.')
        from data_preparer import run_data_preparer
        run_data_preparer(land_marker, dataset_images_dir, dataset_csv)
    else:
        print('[INFO]', f'Dataset file: "{dataset_csv}" found.')

    data = pd.read_csv(dataset_csv)  # Load dataset

    preview_data(data, label='emotion')
    x_train, x_test, y_train, y_test = preprocess_data(data)

    # Create the classifiers
    rf_classifier = RandomForestClassifier(n_estimators=100)
    svm_classifier = svm.SVC()

    print("[Support Vector Machines Accuracy]:")
    evaluate_model(x_train, y_train, x_test, y_test, classifier=svm_classifier)

    print("[Random Forest Accuracy]:")
    evaluate_model(x_train, y_train, x_test, y_test, classifier=rf_classifier)


if __name__ == '__main__':
    run_emotion_detection_evaluator(
        predictor_path='utils/shape_predictor_68_face_landmarks.dat',
        dataset_csv='data/csv/dataset.csv',
        dataset_images_dir='data/raw'
    )
