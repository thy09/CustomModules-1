import pandas as pd
from .utils import (logger, torch_loader, ScoreColumnConstants,
                    _IDENTIFIER_NAME, _LABEL_NAME, generate_score_column_meta)
# from utils import (logger, torch_loader, ScoreColumnConstants,
#                     _IDENTIFIER_NAME, _LABEL_NAME, generate_score_column_meta)
import os
import fire
import torch
import torch.nn as nn
from torchvision import transforms
from azureml.studio.core.io.model_directory import load_model_from_directory
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.data_frame_schema import DataFrameSchema


class Score:
    def __init__(self, model_path, meta={}):
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        self.model, self.id_to_class_dict = load_model_from_directory(
            model_path, model_loader=torch_loader).data
        self.model.eval()

    def run(self, loader_dir, meta=None):
        result_list = []
        for img, label, identifier in loader_dir.iter_images():
            # Convert PIL image to tensor
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
            result_list.append(result)

        # Generate column names
        column_names = [
            _IDENTIFIER_NAME, _LABEL_NAME,
            ScoreColumnConstants.ScoredLabelsColumnName
        ] + [
            f'{ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern}_{self.id_to_class_dict[str(i)]}'
            for i in range(len(self.id_to_class_dict))
        ]
        logger.info(f'schema: {column_names}')
        result_df = pd.DataFrame(result_list, columns=column_names)
        # Remove column _LABEL_NAME if it is None, which means no provided label info
        if result_df[_LABEL_NAME].isnull().any():
            logger.info(
                f"Remove {_LABEL_NAME} because input data is not of 'ImageFolder' type"
            )
            result_df.drop(columns=[_LABEL_NAME], inplace=True)

        return result_df

    def infer(self, data_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        loader_dir = ImageDirectory.load(data_path)
        logger.info(f'Predicting:')
        predict_df = self.run(loader_dir)
        # Generate meta data
        score_columns = generate_score_column_meta(predict_df=predict_df)
        label_column_name = _LABEL_NAME if _LABEL_NAME in predict_df.columns else None
        meta_data = DataFrameSchema(
            column_attributes=DataFrameSchema.generate_column_attributes(
                df=predict_df),
            score_column_names=score_columns,
            label_column_name=label_column_name)
        # Save as data_frame_directory
        save_data_frame_to_directory(save_path,
                                     data=predict_df,
                                     schema=meta_data.to_dict())
        logger.info("DataFrame dumped")


def entrance(model_path='/mnt/chjinche/projects/saved_model',
             data_path='/mnt/chjinche/data/test_data/',
             save_path='/mnt/chjinche/data/scored_nolabel'):
    score = Score(model_path)
    logger.info("model init finished.")
    score.infer(data_path=data_path, save_path=save_path)


if __name__ == '__main__':
    fire.Fire(entrance)
