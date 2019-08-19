from .train import CustomConfig
from .mrcnn.visualize import display_instances
from .mrcnn import model as modellib, utils
from .utils import logger, load_image_folder, compute_batch_ap
from azureml.core.run import Run
import tensorflow as tf
import pickle
import random
import numpy as np
import os
import json
import fire
import base64
import pandas as pd
import pyarrow.parquet as pq


class InferenceConfig(CustomConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNScore:
    def __init__(self, model_folder, meta={}):
        self.config = InferenceConfig()
        self.config.display()
        model_file_name = meta.get('Model file', '')
        model_file_path = os.path.join(model_folder, model_file_name)
        mapping_file_path = os.path.join(model_folder, "mapping.json")
        with tf.device('/gpu:0'):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=model_folder, config=self.config)
            self.model.load_weights(model_file_path, by_name=True)
        with open(mapping_file_path, 'r') as fin:
            self.class_names = json.load(fin)["class_names"]

    def run(self, input_df, meta=None):
        run = Run.get_context()
        img_list = load_image_folder(input_df)
        out_img_str_list = []
        for i, image in enumerate(img_list):
            # compute predictions
            predictions = self.model.detect([image], verbose=1)
            # Display results
            r = predictions[0]
            pred_filename = 'pred_df_{}'.format(i)
            fig = display_instances(image, boxes=r['rois'], masks=r['masks'],
                                    class_ids=r['class_ids'], class_names=self.class_names,
                                    scores=r['scores'], title="Predictions")
            run.log_image("prediction/" + pred_filename, plot=fig)
            fig.savefig('dump.jpg')
            with open('dump.jpg', 'rb') as f:
                out_fig_str = 'data:image/jpg;base64,' + base64.b64encode(f.read()).decode('ascii')
            out_img_str_list.append(out_fig_str)
        df = pd.DataFrame(out_img_str_list, columns=['result'])
        return df

    # def evaluate(self, dataset_score_folder, prediction_folder):
    #     run = Run.get_context()
    #     dataset_score_path = os.path.join(dataset_score_folder, "dataset.pkl")
    #     dataset_score = pickle.load(open(dataset_score_path, 'rb'))
    #     if not dataset_score.image_info[random.choice(dataset_score.image_ids)]["polygons"] is None:
    #         # Metrics
    #         APs = compute_batch_ap(self.model, self.config, dataset_score, dataset_score.image_ids)
    #         with open(os.path.join(prediction_folder, 'metrics.txt'), 'w') as fout:
    #             fout.write("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))
    #     # Predictions
    #     for image_id in dataset_score.image_ids:
    #         image = dataset_score.load_image(image_id)
    #         info = dataset_score.image_info[image_id]
    #         # logger.info("image ID: {}.{} ({}) {}".format(
    #         #     info["source"], info["id"], image_id, dataset_score.image_reference(image_id)))
    #         logger.info("image ID: {}.{} ({})".format(info["source"], info["id"], image_id))
    #         # Run object detection
    #         results = self.model.detect([image], verbose=1)
    #         # Display results
    #         r = results[0]
    #         pred_filename = 'pred_{}'.format(info["id"].split('.jpg')[0])
    #         fig = display_instances(image, boxes=r['rois'], masks=r['masks'],
    #                                 class_ids=r['class_ids'], class_names=self.class_names,
    #                                 scores=r['scores'], title="Predictions")
    #         run.log_image("prediction/" + pred_filename, plot=fig)
    #         fig.savefig(os.path.join(prediction_folder, pred_filename))


def test(model_folder, input_df_folder, dataset_score_folder, prediction_folder, model_filename):
    meta = {'Model file': str(model_filename)}
    maskrcnn = MaskRCNNScore(model_folder, meta=meta)
    input_df_path = os.path.join(input_df_folder, 'data.dataset.parquet')
    if os.path.exists(input_df_path):
        input_df = pd.read_parquet(input_df_path, engine='pyarrow')
        maskrcnn.run(input_df=input_df)


if __name__ == '__main__':
    fire.Fire(test)
