import pandas as pd
import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
from PIL import Image
from io import BytesIO
from .smt_fake import smt_fake_file
from .utils import get_transform, load_model, logger
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.io.data_frame_directory import load_data_frame_from_directory
import base64
import os
import fire
import torch
import torch.nn as nn
import json


class Score:
    def __init__(self, model_path, meta={}):
        _, self.inference_transforms = get_transform()
        self.memory_efficient = True if meta['Memory efficient'] == 'True' else False
        self.model = load_model(model_path, model_type=meta['Model type'], memory_efficient=self.memory_efficient,
                                num_classes=int(meta['Num of classes']))
        self.model.eval()
        with open(os.path.join(model_path, 'index_to_label.json')) as f:
            self.classes = json.load(f)

    def run(self, input, meta=None):
        my_list = []
        for i in range(input.shape[0]):
            temp_string = input.iloc[i]['image_string']
            if temp_string.startswith('data:'):
                my_index = temp_string.find('base64,')
                temp_string = temp_string[my_index+7:]
            temp = base64.b64decode(temp_string)
            img = Image.open(BytesIO(temp))
            input_tensor = self.inference_transforms(img)
            input_tensor = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            with torch.no_grad():
                output = self.model(input_tensor)
                softmax = nn.Softmax(dim=1)
                pred_probs = softmax(output).cpu().numpy()[0]
                index = torch.argmax(output, 1)[0].cpu().item()
            result = [self.classes[index], str(pred_probs[index])]
            my_list.append(result)
        df = pd.DataFrame(my_list, columns=['category', 'probability'])
        return df

    def infer(self, data_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        dir_data = load_data_frame_from_directory(data_path)
        input_df = dir_data.data
        # input_df = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
        df = self.run(input_df)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def entrance(model_path='script/saved_model', data_path='script/outputs', save_path='script/outputs2',
         model_type='densenet201', memory_efficient=False, num_classes=3):
    meta = {'Model type': model_type, 'Memory efficient': str(memory_efficient), 'Num of classes': str(num_classes)}
    score = Score(model_path, meta)
    score.infer(data_path=data_path, save_path=save_path)
    # workaround for smt
    smt_fake_file(save_path)


if __name__ == '__main__':
    fire.Fire(entrance)
