# from torch.utils.cpp_extension import CUDA_HOME
# print(torch.cuda.is_available())
# print(CUDA_HOME)
from PIL import Image
from io import BytesIO
from azureml.core.run import Run
from maskrcnn_benchmark.config import cfg
from .predictor import COCODemo
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import json
import fire
import base64
import pandas as pd


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


def imshow(img, out_folder, out_filename):
    run = Run.get_context()
    img_plt = plt.figure(1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    # plt.show()
    run.log_image("prediction/"+out_filename, plot=img_plt)
    out_fig_path = os.path.join(out_folder, out_filename)
    img_plt.savefig(out_fig_path)
    with open(out_fig_path, 'rb') as f:
        out_fig_str = 'data:image/jpg;base64,'+base64.b64encode(f.read()).decode('ascii')
    return out_fig_str


class MaskRCNNBenchMark:
    def __init__(self, model_folder, meta={}):
        # this makes our figures bigger
        pylab.rcParams['figure.figsize'] = 20, 12
        config_file_name = meta.get('Config file', '')
        cfg.merge_from_file(os.path.join(model_folder, config_file_name))
        # manual override some options
        # only "cuda" and "cpu" are valid device types
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

    def run(self, test_folder, prediction_folder, meta=None):
        os.makedirs(prediction_folder, exist_ok=True)
        input_df = pd.read_parquet(os.path.join(test_folder, 'data.dataset.parquet'), engine='pyarrow')
        img_list = load_image_folder(input_df)
        out_img_str_list = []
        for i, image in enumerate(img_list):
            # compute predictions
            predictions = self.coco_demo.run_on_opencv_image(image)
            out_img_str_list.append((predictions, prediction_folder, 'result_{}.jpg'.format(i)))
        df = pd.DataFrame(out_img_str_list, columns=['result'])
        return df


def test(model_folder, test_folder, prediction_folder):
    maskrcnn = MaskRCNNBenchMark(model_folder)
    maskrcnn.run(test_folder=test_folder, prediction_folder=prediction_folder)

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
