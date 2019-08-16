from PIL import Image
from io import BytesIO
from .visualize import display_instances
from .mrcnn import model as modellib
from .mrcnn.config import Config

import numpy as np
import os
import json
import fire
import base64
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from azureml.core.run import Run


class InferenceConfig(Config):
    # Run detection on one image at a time
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def load_image_folder(input_df):
    img_list = []
    for i in range(input_df.shape[0]):
        temp_string = input_df.iloc[i]['image_string']
        if temp_string.startswith('data:'):
            my_index = temp_string.find('base64,')
            temp_string = temp_string[my_index + 7:]
        temp = base64.b64decode(temp_string)
        pil_img = Image.open(BytesIO(temp)).convert("RGB")
        # convert to BGR format
        img_list.append(np.array(pil_img)[:, :, [2, 1, 0]])
    return img_list


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class MaskRCNN:
    def __init__(self, model_folder, meta={}):
        config = InferenceConfig()
        config.display()
        model_file_name = meta.get('Model file', '')
        print(os.path.join(model_folder, model_file_name))
        with tf.device('/cpu:0'):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=model_folder, config=config)
            self.model.load_weights(os.path.join(model_folder, model_file_name), by_name=True)

    def run(self, test_folder, meta=None):
        run = Run.get_context()
        input_df = pd.read_parquet(os.path.join(test_folder, 'data.dataset.parquet'), engine='pyarrow')
        img_list = load_image_folder(input_df)
        out_img_str_list = []
        for i, image in enumerate(img_list):
            # compute predictions
            predictions = self.model.detect([image], verbose=1)
            # Display results
            r = predictions[0]
            pred_filename = 'pred_{}'.format(i)
            fig = display_instances(image, boxes=r['rois'],
                              masks=r['masks'], class_ids=r['class_ids'], class_names=class_names,
                              scores=r['scores'],
                              title="Predictions")
            run.log_image("prediction/" + pred_filename, plot=fig)
            fig.savefig('dump.jpg')
            with open('dump.jpg', 'rb') as f:
                out_fig_str = 'data:image/jpg;base64,' + base64.b64encode(f.read()).decode('ascii')
            out_img_str_list.append(out_fig_str)
        df = pd.DataFrame(out_img_str_list, columns=['result'])
        return df


def test(model_folder, test_folder, prediction_folder, model_filename):
    meta = {'Model file': str(model_filename)}
    maskrcnn = MaskRCNN(model_folder, meta=meta)
    maskrcnn.run(test_folder=test_folder)

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'Dataset',
        'Name': 'Dataset .NET file',
        'ShortName': 'Dataset',
        'Description': 'A serialized DataTable supporting partial reads and writes',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'dataset.parquet',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': True,
        'AllowModelPromotion': False,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(prediction_folder, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


if __name__ == '__main__':
    fire.Fire(test)
