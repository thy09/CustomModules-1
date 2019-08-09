from .arg_opts import score_opts, logger
from .train import CustomConfig
from .visualize import display_instances
import os
import numpy as np
import tensorflow as tf
import pickle
import random
from mrcnn import model as modellib, utils
from preprocess import CustomDataset


class InferenceConfig(CustomConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def compute_batch_ap(model, config, dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            meta = modellib.parse_image_meta(image_meta[np.newaxis, ...])
            logger.info("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs


def main():
    parser = score_opts()
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.prediction_folder):
        os.makedirs(args.prediction_folder)
    # Configurations
    config = InferenceConfig()
    config.display()
    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=args.model_folder, config=config)

    # weights_path = "/path/to/mask_rcnn_balloon.h5"
    # Or, load the last model you trained
    weights_path = model.find_last()
    # Load weights
    logger.info("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    dataset_score_path = os.path.join(args.dataset_score_folder, args.dataset_score_file)
    dataset_score = pickle.load(open(dataset_score_path, 'rb'))
    if not dataset_score.image_info[random.choice(dataset_score.image_ids)]["polygons"] is None:
        # Metrics
        APs = compute_batch_ap(model, config, dataset_score, dataset_score.image_ids)
        with open(os.path.join(args.prediction_folder, 'metrics.txt'), 'w') as fout:
            fout.write("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))
    # Predictions
    for image_id in dataset_score.image_ids:
        image = dataset_score.load_image(image_id)
        info = dataset_score.image_info[image_id]
        # logger.info("image ID: {}.{} ({}) {}".format(
        #     info["source"], info["id"], image_id, dataset_score.image_reference(image_id)))
        logger.info("image ID: {}.{} ({})".format(info["source"], info["id"], image_id))
        # Run object detection
        results = model.detect([image], verbose=1)
        # Display results
        r = results[0]
        pred_filename = 'pred_{}'.format(info["id"].split('.jpg')[0])
        display_instances(image, out_folder=args.prediction_folder, out_filename=pred_filename, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'], class_names=dataset_score.class_names, scores=r['scores'],
                          title="Predictions")


if __name__ == '__main__':
    main()
