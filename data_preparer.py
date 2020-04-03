import enum
from os import listdir, rename, remove
from shutil import rmtree

FILE_SEPARATOR = '/'  # For Linux and MacOS
LABEL_POSTFIX = '_emotion.txt'


def dir_content(dir_path: str):
    return sorted(listdir(dir_path))


def merge_paths(*paths: str):
    return FILE_SEPARATOR.join(paths).replace('//', '/')


def list_to_csv_line(lst: []):
    return ','.join(map(str, lst)) + '\n'


class EmotionLabels(enum.Enum):
    neutral = {'code': '   0.0000000e+00', 'name': 'neutral'}
    anger = {'code': '   1.0000000e+00', 'name': 'anger'}
    contempt = {'code': '   2.0000000e+00', 'name': 'contempt'}
    disgust = {'code': '   3.0000000e+00', 'name': 'disgust'}
    fear = {'code': '   4.0000000e+00', 'name': 'fear'}
    happy = {'code': '   5.0000000e+00', 'name': 'happy'}
    sadness = {'code': '   6.0000000e+00', 'name': 'sadness'}
    surprise = {'code': '   7.0000000e+00', 'name': 'surprise'}

    @staticmethod
    def code_to_name(input_label_code: str):
        matched_labels = [label for label in EmotionLabels if label.value['code'] == input_label_code]
        return matched_labels[0].value['name']


class ImageDataset:
    img_extension = '.png'
    label_extension = '_emotion.txt'

    def __init__(self, path: str, subject: str, emotion_num):
        self.path = path
        self.subject = subject
        self.emotion_num = emotion_num

    @property
    def img_paths(self) -> []:
        img_names = [img for img in dir_content(self.path) if img.endswith(self.img_extension)]
        return [merge_paths(self.path, img_name) for img_name in img_names]

    @property
    def img_paths_with_labels(self) -> [dict]:
        return [
            {'path': self.img_paths[0], 'label': EmotionLabels.neutral.name},
            {'path': self.img_paths[-1], 'label': self.emotion_label}
        ]

    @property
    def emotion_label(self) -> str:
        label_file = [img for img in dir_content(self.path) if img.endswith(self.label_extension)][0]
        label_path = merge_paths(self.path, label_file)
        encoded_label = open(label_path, 'r').read().replace('\n', '')
        return EmotionLabels.code_to_name(input_label_code=encoded_label)

    def regulate_names(self):
        for count, img_path in enumerate(self.img_paths):
            formatted_cnt = str(count + 1).zfill(8)
            new_img_name = '%s_%s_%s.png' % (self.subject, self.emotion_num, formatted_cnt)
            rename(src=img_path, dst=merge_paths(self.path, new_img_name))

    def remove_except_first_and_last(self):
        for pic in self.img_paths[1:-1]:
            remove(pic)

    def clear_img_paths_without_label(self, label_postfix: str):
        for img_path in self.img_paths:
            if not any(label_postfix in pic for pic in dir_content(img_path)):
                rmtree(img_path)

    @staticmethod
    def collect_img_datasets(dataset_parent_dir: str) -> ['ImageDataset']:
        image_datasets = []
        for subject in dir_content(dataset_parent_dir):
            subject_dir = merge_paths(dataset_parent_dir, subject)
            for emotion_num in dir_content(subject_dir):
                path = merge_paths(subject_dir, emotion_num)
                image_datasets.append(ImageDataset(path=path, subject=subject, emotion_num=emotion_num))
        return image_datasets


class PreProcessor:
    def __init__(self, data_dir: str):
        self.img_datasets = ImageDataset.collect_img_datasets(data_dir)

    def preprocess(self):
        for img_ds in self.img_datasets:
            img_ds.regulate_names()
            img_ds.remove_except_first_and_last()
            img_ds.clear_img_paths_without_label(label_postfix=LABEL_POSTFIX)


class DatasetBuilder:
    def __init__(self, data_dir: str, class_feature: str, land_marker: LandMarker):
        self.img_datasets = ImageDataset.collect_img_datasets(data_dir)
        self.land_marker = land_marker
        self.header = self.create_header(class_feature=class_feature)

    def create_header(self, class_feature: str):
        lm = self.land_marker
        landmark_points = lm.img_to_landmarks(img_path=self.extract_imgs_with_labels()[0]['path'])
        header_list = tuple('X%d' % (i + 1) for i in range(len(landmark_points))) + (class_feature,)
        return list_to_csv_line(header_list)

    def build(self, target: str, write_header: bool = True):
        lm = self.land_marker

        imgs_w_labels = self.extract_imgs_with_labels()

        print('[LOG]', 'Dataset is building..')
        with open(file=target, mode='w') as csv_dataset:
            if write_header:
                csv_dataset.write(self.header)
            for i, img_w_label in enumerate(imgs_w_labels):
                landmark_points = lm.img_to_landmarks(img_path=img_w_label['path'])
                instance = landmark_points + (img_w_label['label'],)
                csv_dataset.write(list_to_csv_line(instance))
                print('[LOG]', 'Written Instance Progress: %d/%d' % ((i + 1), len(imgs_w_labels)))
        print('\nAll instances are successfully written to file: \"%s\"' % target)

    def extract_imgs_with_labels(self) -> [dict]:
        all_imgs_with_labels = []
        for img_ds in self.img_datasets:
            all_imgs_with_labels.extend(img_ds.img_paths_with_labels)
        return all_imgs_with_labels
