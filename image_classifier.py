from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils.data_land_marker import LandMarker


class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.class_feature = df.columns.values[-1:][0]  # class_feature = last feature
        self.input_features = list(df.columns.values[:-1])  # class_feature = features except last feature
        self.unique_labels = pd.factorize(self.labels)[1]

    @classmethod
    def from_csv(cls, file_path: str): return cls(df=pd.read_csv(file_path))

    @property
    def labels(self) -> []: return self.data[self.class_feature]

    @property
    def factorized_labels(self) -> []: return pd.factorize(self.labels)[0]

    @property
    def without_labels(self) -> pd.DataFrame: return self.data[self.input_features]

    @property
    def instance_count(self) -> int: return len(self.data)


class Classifier:
    def __init__(self, csv_path: str, algorithm: str, land_marker: LandMarker):
        ds = Dataset.from_csv(file_path=csv_path)
        self.landmarker = land_marker
        self.dataset = ds

        if algorithm == 'RandomForest':
            self.classifier = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
            self.classifier.fit(ds.without_labels, ds.factorized_labels)
        elif algorithm == 'SVM':
            self.classifier = SVC(kernel='linear')
            self.classifier.fit(ds.without_labels, ds.factorized_labels)
        else:
            raise ValueError('%s algorithm is not defined' % algorithm)

        # classifier.evaluate_with_random_forest(split_ratio=0.75)

    def classify(self, img: np.ndarray) -> List[str]:
        face_land_marks_list = self.landmarker.img_to_landmarks(img, exclude_vector_base=True)
        if not face_land_marks_list:
            return ['no face']
        predicted_labels = []
        for face_land_marks in face_land_marks_list:
            predicted_class_idx = self.classifier.predict(X=[face_land_marks])
            predicted_classes = self.dataset.unique_labels[predicted_class_idx]
            predicted_labels.append(predicted_classes[0])
        return predicted_labels

    def extract_face_rectangle(self, img: np.ndarray) -> List[Tuple]:
        return self.landmarker.img_to_rectangles(img=img)

    def extract_landmark_points(self, img: np.ndarray) -> List[np.ndarray]:
        return self.landmarker.img_to_landmark_points(img)

    def evaluate_with_random_forest(self, split_ratio: float, n_jobs: int = 2, random_state: int = 0):
        a_classifier = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state, n_estimators=100)

        train, test = self.split_data(split_ratio=split_ratio)  # type: Dataset
        a_classifier.fit(train.without_labels, train.factorized_labels)

        predicted_labels = [test.unique_labels[predict] for predict in a_classifier.predict(X=test.without_labels)]

        match_counter = 0
        for real, predicted in zip(test.labels, predicted_labels):
            print('> REAL:"%s"\t\tPREDICTED:"%s"' % (real, predicted))
            if real == predicted:
                match_counter += 1

        print('Accuracy: %s %%' % (100 * match_counter / test.instance_count))

    def split_data(self, split_ratio: float) -> tuple:
        is_train_feature = 'is_train'
        data = self.dataset.data

        np.random.seed(0)
        data[is_train_feature] = np.random.uniform(0, 1, len(data)) <= split_ratio
        train_df, test_df = data[data[is_train_feature] == True], data[data[is_train_feature] == False]
        del train_df[is_train_feature], test_df[is_train_feature]

        print('[INFO]', 'Size of training data: %d, Size of test data: %d' % (len(train_df), len(test_df)))
        return Dataset(train_df), Dataset(test_df)
