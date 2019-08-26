import pandas as pd
from PIL import Image
from io import BytesIO
from .utils import get_transform, load_model, logger
from azureml.studio.common.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
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
        df = self.run(input_df)
        schema = {
            'columnAttributes': [
                {
                    "name": "category",
                    "type": "String",
                    "isFeature": True,
                    "elementType": {
                        "typeName": "str",
                        "isNullable": False
                    },
                },
                {
                    "name": "probability",
                    "type": "String",
                    "isFeature": True,
                    "elementType": {
                        "typeName": "str",
                        "isNullable": False
                    },
                },
            ],
            "featureChannels": [],
            "labelColumns": {},
            "scoreColumns": {},
        }
        save_data_frame_to_directory(save_path, data=df, schema=schema)
        print(f"DataFrame dumped: {df}")


def entrance(model_path='script/saved_model', data_path='script/outputs', save_path='script/outputs2',
         model_type='densenet201', memory_efficient=False, num_classes=3):
    meta = {'Model type': model_type, 'Memory efficient': str(memory_efficient), 'Num of classes': str(num_classes)}
    score = Score(model_path, meta)
    score.infer(data_path=data_path, save_path=save_path)


if __name__ == '__main__':
    fire.Fire(entrance)
