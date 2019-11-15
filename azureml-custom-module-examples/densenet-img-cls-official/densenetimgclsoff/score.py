import pandas as pd
import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
from PIL import Image
from io import BytesIO
from .utils import get_transform, logger, torch_loader
# from utils import get_transform, logger, torch_loader
import base64
import os
import fire
import torch
import torch.nn as nn
from azureml.studio.core.io.model_directory import load_model_from_directory
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.io.image_directory import ImageDirectory


def decode_image_str(input_str):
    if input_str.startswith('data:'):
        start_index = input_str.find('base64,') + 7
        input_str = input_str[start_index:]
    img = Image.open(BytesIO(base64.b64decode(input_str)))
    return img


class Score:
    def __init__(self, model_path, meta={}):
        _, self.inference_transforms = get_transform()
        self.model, self.id_to_class_dict = load_model_from_directory(
            model_path, model_loader=torch_loader).data
        self.model.eval()

    def run(self, loader_dir, meta=None):
        my_list = []
        for img, _, identifier in loader_dir.iter_images():
            # print(img)
            try:
                input_tensor = self.inference_transforms(img)
                input_tensor = input_tensor.unsqueeze(0)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                with torch.no_grad():
                    output = self.model(input_tensor)
                    softmax = nn.Softmax(dim=1)
                    pred_probs = softmax(output).cpu().numpy()[0]
                    index = torch.argmax(output, 1)[0].cpu().item()
                result = [
                    self.id_to_class_dict[str(index)], pred_probs[index],
                    identifier
                ]
                # print(result)
                my_list.append(result)
            except Exception:
                print(f'Exception {identifier}')
                pass
        # print(my_list)
        df = pd.DataFrame(my_list,
                          columns=['category', 'probability', 'identifier'])
        return df

    def infer(self, data_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        loader_dir = ImageDirectory.load(data_path)
        pred_df = self.run(loader_dir)
        save_data_frame_to_directory(save_path, data=pred_df)
        logger.info("DataFrame dumped")


def entrance(model_path='/mnt/chjinche/projects/saved_model',
             data_path='/mnt/chjinche/data/small/',
             save_path='/mnt/chjinche/data/scored'):
    score = Score(model_path)
    score.infer(data_path=data_path, save_path=save_path)


if __name__ == '__main__':
    # workaround for import packages without explicit use
    logger.info(f"Load pyarrow.parquet explicitly: {pq}")
    fire.Fire(entrance)
