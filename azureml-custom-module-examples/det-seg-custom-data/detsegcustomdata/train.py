from .arg_opts import train_opts, logger
from .preprocess import *
import os
import pickle
from mrcnn.config import Config
from mrcnn import model as modellib


parser = train_opts()
args, _ = parser.parse_known_args()


class CustomConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # def __init__(self, name, gpu_cnt=2, num_classes=1, step_per_epoch=100, det_min_conf=0.9):
    #     super(CustomConfig, self).__init__()
    #     self.NAME = name
    #     self.GPU_COUNT = gpu_cnt
    #     # Number of classes (including background)
    #     self.NUM_CLASSES = (1 + num_classes)
    #     # Number of training steps per epoch    
    #     self.STEPS_PER_EPOCH = step_per_epoch
    #     # Skip detections with < 90% confidence
    #     self.DETECTION_MIN_CONFIDENCE = det_min_conf
    NAME = name
    GPU_COUNT = args.gpu_cnt
    # Number of classes (including background)
    NUM_CLASSES = (1 + num_classes)
    # Number of training steps per epoch
    STEPS_PER_EPOCH = args.step_per_epoch
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = args.det_min_conf


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


def load_pretrained_model(model, pretrained_model_path):
    """Load pretrained model."""
    # pretrained_model_path = os.path.join(pretrained_model_folder, pretrained_model_file)
    # Load pretrained weights
    logger.info("Loading pretrained model {}".format(pretrained_model_path))
    if pretrained_model_path.lower().endswith("mask_rcnn_coco.h5"):
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(pretrained_model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(pretrained_model_path, by_name=True)
    return model


def main():
    # parser = train_opts()
    # args, _ = parser.parse_known_args()
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    # Configurations
    # config = CustomConfig(name=args.name, gpu_cnt=args.gpu_cnt, num_classes=args.num_classes, step_per_epoch=args.step_per_epoch)
    config = CustomConfig()
    config.display()
    # Create model
    untrained_model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.model_folder)
    # Load pretrained model
    pretrained_model_path = os.path.join(args.pretrained_model_folder, args.pretrained_model_file)
    pretrained_model = load_pretrained_model(untrained_model, pretrained_model_path)
    # Train
    dataset_train_path = os.path.join(args.dataset_train_folder, args.dataset_train_file)
    dataset_val_path = os.path.join(args.dataset_val_folder, args.dataset_val_file)
    train(pretrained_model, dataset_train_path, dataset_val_path, args.epochs, args.lr)


if __name__ == '__main__':
    main()
