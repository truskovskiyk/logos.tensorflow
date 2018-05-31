import io
import hashlib
from pathlib import Path

import PIL
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def read_image(image_path: Path):
    with tf.gfile.GFile(str(image_path), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG {}'.format(image_path))
    key = hashlib.sha256(encoded_jpg).hexdigest()
    return image, key, encoded_jpg


def create_labels(objects: pd.DataFrame, image: PIL.Image, label_map_dict: dict):
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []
    for _, row in objects.iterrows():

        class_name = row['class']
        xmin = float(row['x1']) / image.width
        xmax = float(row['x2']) / image.width
        ymin = float(row['y1']) / image.height
        ymax = float(row['y2']) / image.height

        if not xmin < xmax or not ymin < ymax:
            print("skip")
            continue

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    return xmins, ymins, xmaxs, ymaxs, classes_text, classes


def create_one_records(image_path: Path, objects: pd.DataFrame, label_map_dict: dict) -> tf.train.Example:
    image, key, encoded_jpg = read_image(image_path)
    xmins, ymins, xmaxs, ymaxs, classes_text, classes = create_labels(objects, image, label_map_dict)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image.height),
        'image/width': dataset_util.int64_feature(image.width),
        'image/filename': dataset_util.bytes_feature(str(image_path).encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(str(image_path).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return example


def create_tf_records(path_to_csv: Path, path_to_images: Path, tf_record_path: Path, label_map_dict: dict):
    df = pd.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    writer = tf.python_io.TFRecordWriter(str(tf_record_path))
    for filename, group in tqdm(df.groupby('filename')):
        tf_example = create_one_records(image_path=path_to_images / filename,
                                        objects=group,
                                        label_map_dict=label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    data_path = Path('./data/')
    dataset_path = Path('./dataset')
    label_map_path = Path('./model_configs/label_map.pbtxt')
    label_map_dict = label_map_util.get_label_map_dict(str(label_map_path))

    create_tf_records(path_to_csv=dataset_path / 'train.csv',
                      path_to_images=data_path / 'flickr_logos_27_dataset_images',
                      tf_record_path=dataset_path / 'train.record',
                      label_map_dict=label_map_dict)

    create_tf_records(path_to_csv=dataset_path / 'val.csv',
                      path_to_images=data_path / 'flickr_logos_27_dataset_images',
                      tf_record_path=dataset_path / 'val.record',
                      label_map_dict=label_map_dict)

    create_tf_records(path_to_csv=dataset_path / 'test.csv',
                      path_to_images=data_path / 'flickr_logos_27_dataset_images',
                      tf_record_path=dataset_path / 'test.record',
                      label_map_dict=label_map_dict)
