from PIL import Image
from io import BytesIO
from .mrcnn import model as modellib, utils
import logging
import base64
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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