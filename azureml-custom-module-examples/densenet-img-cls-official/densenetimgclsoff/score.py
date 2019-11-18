import pandas as pd
import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
from .utils import get_transform, logger, torch_loader
# from utils import get_transform, logger, torch_loader
import os
import fire
import torch
import torch.nn as nn
from torchvision import transforms
from azureml.studio.core.io.model_directory import load_model_from_directory
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.modules.ml.common.ml_utils import generate_score_column_meta, TaskType
from azureml.studio.modules.ml.common.constants import ScoreColumnConstants
from azureml.studio.common.datatable.data_table import DataTable


class Score:
    def __init__(self, model_path, meta={}):
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        self.model, self.id_to_class_dict = load_model_from_directory(
            model_path, model_loader=torch_loader).data
        self.model.eval()

    def run(self, loader_dir, meta=None):
        my_list = []
        for img, label, identifier in loader_dir.iter_images():
            # print(f'label: {label}')
            input_tensor = self.to_tensor_transform(img)
            input_tensor = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            with torch.no_grad():
                output = self.model(input_tensor)
                softmax = nn.Softmax(dim=1)
                pred_probs = softmax(output).cpu().numpy()[0]
            index = torch.argmax(output, 1)[0].cpu().item()
            result = [identifier, label, self.id_to_class_dict[str(index)]
                      ] + list(pred_probs)
            my_list.append(result)

        schema_columns = [
            'identifier', 'label', ScoreColumnConstants.ScoredLabelsColumnName
        ] + [
            f'{ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern}_{self.id_to_class_dict[str(i)]}'
            for i in range(len(self.id_to_class_dict))
        ]
        logger.info(f'schema: {schema_columns}')
        df = pd.DataFrame(my_list, columns=schema_columns)
        if df['label'].isnull().any():
            logger.info("Remove label because input data is not of 'ImageFolder' type")
            df.drop(columns=['label'], inplace=True)
        return df

    def infer(self, data_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        loader_dir = ImageDirectory.load(data_path)
        logger.info(f'Predicting:')
        pred_df = self.run(loader_dir)
        score_columns = generate_score_column_meta(
            task_type=TaskType.MultiClassification, predict_df=pred_df)
        pred_dt = DataTable(pred_df)
        pred_dt.meta_data.score_column_names = score_columns
        save_data_frame_to_directory(save_path, data=pred_dt.data_frame, schema=pred_dt.meta_data.to_dict())
        logger.info("DataFrame dumped")


def entrance(model_path='/mnt/chjinche/projects/saved_model',
             data_path='/mnt/chjinche/data/test_data/',
             save_path='/mnt/chjinche/data/scored_nolabel'):
    score = Score(model_path)
    logger.info("model init finished.")
    score.infer(data_path=data_path, save_path=save_path)


if __name__ == '__main__':
    # workaround for import packages without explicit use
    logger.info(f"Load pyarrow.parquet explicitly: {pq}")
    fire.Fire(entrance)
