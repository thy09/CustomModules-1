from PIL import Image
from azureml.core.run import Run
from maskrcnn_benchmark.config import cfg
from .predictor import COCODemo

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--model_folder', default='./model_folder', help='model folder path')
    # parser.add_argument('--checkpoint_filename', default='', help='checkpoint filename')
    parser.add_argument('--config_filename', default='', help='config filename')
    parser.add_argument('--test_folder', default='./test_images', help='test image folder')
    parser.add_argument('--prediction_folder', help='output result file')
    args = parser.parse_args()
    return args


def load_image_folder(img_folder):
    img_list = []
    for img_name in os.listdir(img_folder):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(img_folder, img_name)
            pil_img = Image.open(img_path).convert("RGB")
            # convert to BGR format
            img_list.append(np.array(pil_img)[:, :, [2, 1, 0]])
    return img_list


def imshow(img, out_folder, out_filename):
    run = Run.get_context()
    img_plt = plt.figure(1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    # plt.show()
    run.log_image("prediction/"+out_filename, plot=img_plt)
    img_plt.savefig(os.path.join(out_folder, out_filename))


def main():
    args = parse_args()
    if not os.path.exists(args.prediction_folder):
        os.makedirs(args.prediction_folder)
    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12
    config_file = os.path.join(args.model_folder, args.config_filename)
    cfg.merge_from_file(config_file)
    # manual override some options
    # only "cuda" and "cpu" are valid device types
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    img_list = load_image_folder(args.test_folder)
    for i, image in enumerate(img_list):
        # compute predictions
        predictions = coco_demo.run_on_opencv_image(image)
        imshow(predictions, args.prediction_folder, 'result_{}.jpg'.format(i))


if __name__ == '__main__':
    main()
