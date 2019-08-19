from .utils import logger, load_pretrained_model
from .mrcnn.config import Config
from .mrcnn import model as modellib
from shutil import copyfile
import os
import pickle
import fire
import json


class CustomConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    def set_attr(self, gpu_cnt=2, nc=1, step_per_epoch=100, det_min_conf=0.9):
        CustomConfig.NAME = "custom"
        CustomConfig.GPU_COUNT = gpu_cnt
        # Number of classes (including background)
        CustomConfig.NUM_CLASSES = nc
        # Number of training steps per epoch
        CustomConfig.STEPS_PER_EPOCH = step_per_epoch
        # Skip detections with < 90% confidence
        CustomConfig.DETECTION_MIN_CONFIDENCE = det_min_conf


def train(model, dataset_train_path, dataset_val_path, epochs=30, lr=0.001):
    """Train the model."""
    # Training dataset
    logger.info("loading train dataset.")
    dataset_train = pickle.load(open(dataset_train_path, 'rb'))
    # Validation dataset
    logger.info("loading validation dataset.")
    dataset_val = pickle.load(open(dataset_val_path, 'rb'))
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    logger.info("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=lr,
                epochs=epochs,
                layers='heads')
    logger.info("Finished training.")


def test(pretrained_model_folder, pretrained_model_file, dataset_train_folder, dataset_val_folder, model_folder,
         gpu_cnt, epochs, lr):
    os.makedirs(model_folder, exist_ok=True)
    mapping_file_path = os.path.join(dataset_train_folder, 'mapping.json')
    with open(mapping_file_path, 'r') as fin:
        num_classes = json.load(fin)["class_names"]
    # Configurations
    config = CustomConfig()
    config.set_attr(
        gpu_cnt=gpu_cnt,
        nc=num_classes
    )
    config.display()
    # Create model
    untrained_model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_folder)
    # Load pretrained model
    pretrained_model_path = os.path.join(pretrained_model_folder, pretrained_model_file)
    pretrained_model = load_pretrained_model(untrained_model, pretrained_model_path)
    # Train
    dataset_train_path = os.path.join(dataset_train_folder, "dataset.pkl")
    dataset_val_path = os.path.join(dataset_val_folder, "dataset.pkl")
    train(pretrained_model, dataset_train_path, dataset_val_path, epochs, lr)
    copyfile(os.path.join(dataset_train_folder, 'mapping.json'), os.path.join(model_folder, 'mapping.json'))


if __name__ == '__main__':
    fire.Fire(test)
