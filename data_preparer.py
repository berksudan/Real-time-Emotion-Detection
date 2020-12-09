import enum
from os import listdir
from typing import List, Dict, Optional, Iterable

from utils.data_land_marker import LandMarker

FILE_SEPARATOR = '/'  # For Linux and MacOS
LABEL_POSTFIX = '_emotion.txt'


def sorted_dir_content(dir_path: str):
    return sorted(listdir(dir_path))


def merge_paths(*paths: str):
    return FILE_SEPARATOR.join(paths).replace('//', '/')


def list_to_csv_line(lst: Iterable):
    return ','.join(map(str, lst)) + '\n'


class EmotionLabels(enum.Enum):
    neutral = {'code': '0.0000000e+00', 'name': 'neutral'}
    anger = {'code': '1.0000000e+00', 'name': 'anger'}
    contempt = {'code': '2.0000000e+00', 'name': 'contempt'}
    disgust = {'code': '3.0000000e+00', 'name': 'disgust'}
    fear = {'code': '4.0000000e+00', 'name': 'fear'}
    happy = {'code': '5.0000000e+00', 'name': 'happy'}
    sadness = {'code': '6.0000000e+00', 'name': 'sadness'}
    surprise = {'code': '7.0000000e+00', 'name': 'surprise'}

    @staticmethod
    def code_to_name(input_label_code: str) -> str:
        matched_labels = [label for label in EmotionLabels if label.value['code'] == input_label_code]
        return matched_labels[0].value['name']


class EmotionFrameSet:
    IMG_EXTENSION = '.png'
    LABEL_EXTENSION = '_emotion.txt'

    class LabeledImage:
        def __init__(self, img_path: str, img_label: str):
            self.img_path = img_path
            self.img_label = img_label

        def __str__(self):
            return str({"IMG_PATH": self.img_path, "IMG_LABEL": self.img_label})

    @staticmethod
    def print_labeled_images(labeled_images: List[LabeledImage]) -> None:
        for i, lbl_img in enumerate(labeled_images):
            print(f'[{i}]:', lbl_img)

    def __init__(self, emotion_frame_set_dir_path: str):
        files_in_dir = sorted_dir_content(dir_path=emotion_frame_set_dir_path)
        self.img_paths = self.calc_img_paths(dir_path=emotion_frame_set_dir_path, files_in_dir=files_in_dir)
        self.emotion_label = self.calc_emotion_label(dir_path=emotion_frame_set_dir_path, files_in_dir=files_in_dir)

    @staticmethod
    def calc_emotion_label(dir_path: str, files_in_dir: List[str]) -> Optional[str]:
        possible_label_files = [file for file in files_in_dir if file.endswith(EmotionFrameSet.LABEL_EXTENSION)]
        if not possible_label_files:  # Check if emotion label does not exist.
            return None
        label_file = possible_label_files[0]
        label_path = merge_paths(dir_path, label_file)
        encoded_label = open(label_path, 'r').read().strip()
        return EmotionLabels.code_to_name(input_label_code=encoded_label)

    @staticmethod
    def calc_img_paths(dir_path: str, files_in_dir: List[str]) -> List:
        img_names = [img for img in files_in_dir if img.endswith(EmotionFrameSet.IMG_EXTENSION)]
        return [merge_paths(dir_path, img_name) for img_name in img_names]

    @staticmethod
    def calc_img_paths_with_labels(self) -> List[Dict]:
        return [
            {'path': self.img_paths[0], 'label': EmotionLabels.neutral.name},
            {'path': self.img_paths[-1], 'label': self.emotion_label}
        ]


class PreProcessor:
    def __init__(self, dataset_parent_dir: str):
        self.__dataset_parent_dir = dataset_parent_dir

    def preprocess(self) -> List[EmotionFrameSet.LabeledImage]:
        emotion_frame_sets = self.__collect_emotion_frame_sets(dataset_parent_dir=self.__dataset_parent_dir)
        filtered_emotion_frame_sets = self.__filter_emotion_frame_sets(emotion_frame_sets=emotion_frame_sets)
        labeled_images = self.__emotion_frame_sets_to_labeled_images(frame_sets=filtered_emotion_frame_sets)
        # EmotionFrameSet.print_labeled_images(labeled_images)
        return labeled_images

    @staticmethod
    def __collect_emotion_frame_sets(dataset_parent_dir: str) -> List[EmotionFrameSet]:
        emotion_frame_sets = []  # type: List[EmotionFrameSet]
        for person in sorted_dir_content(dataset_parent_dir):
            person_subject_dir = merge_paths(dataset_parent_dir, person)
            for emotion_folder in sorted_dir_content(person_subject_dir):
                path = merge_paths(person_subject_dir, emotion_folder)
                emotion_frame_sets.append(EmotionFrameSet(emotion_frame_set_dir_path=path))
        return emotion_frame_sets

    @staticmethod
    def __filter_emotion_frame_sets(emotion_frame_sets: List[EmotionFrameSet]) -> List[EmotionFrameSet]:
        filtered_emotion_frame_sets = []  # type: # type: List[EmotionFrameSet]
        for emotion_frame_set in emotion_frame_sets:
            if emotion_frame_set.emotion_label is not None:
                all_img_paths = emotion_frame_set.img_paths
                emotion_frame_set.img_paths = [all_img_paths[0], all_img_paths[-1]]  # Get first and last image.
                filtered_emotion_frame_sets.append(emotion_frame_set)
        return filtered_emotion_frame_sets

    @staticmethod
    def __emotion_frame_sets_to_labeled_images(frame_sets: List[EmotionFrameSet]) -> List[EmotionFrameSet.LabeledImage]:
        labeled_images = []  # type: List[EmotionFrameSet.LabeledImage]
        for a_set in frame_sets:
            i1 = EmotionFrameSet.LabeledImage(a_set.img_paths[0], img_label=EmotionLabels.neutral.name)
            i2 = EmotionFrameSet.LabeledImage(a_set.img_paths[1], img_label=a_set.emotion_label)
            labeled_images.append(i1)
            labeled_images.append(i2)
        return labeled_images


class DatasetBuilder:
    def __init__(self, labeled_images: List[EmotionFrameSet.LabeledImage], class_col: str, land_marker: LandMarker):
        self.labeled_images = labeled_images
        self.land_marker = land_marker
        self.header = self.create_header(class_col=class_col, dummy_labeled_image=labeled_images[0])

    def create_header(self, class_col: str, dummy_labeled_image: EmotionFrameSet.LabeledImage):
        lm = self.land_marker
        dummy_lm_points = lm.img_path_to_landmarks(img_path=dummy_labeled_image.img_path)[0]
        header_list = tuple('X%d' % (i + 1) for i in range(len(dummy_lm_points))) + (class_col,)
        return list_to_csv_line(header_list)

    def build(self, target: str, write_header: bool = True):
        lm = self.land_marker

        print('[INFO]', 'Dataset is building..')
        with open(file=target, mode='w') as csv_dataset:
            if write_header:
                csv_dataset.write(self.header)
            for i, labeled_img in enumerate(self.labeled_images):
                landmark_points = lm.img_path_to_landmarks(img_path=labeled_img.img_path)[0]
                print(">", len(landmark_points))

                instance = landmark_points + (labeled_img.img_label,)
                csv_dataset.write(list_to_csv_line(instance))
                print('[INFO]', 'Written Instance Progress: %d/%d' % ((i + 1), len(self.labeled_images)))
        print('\nAll instances are successfully written to file: \"%s\"' % target)
