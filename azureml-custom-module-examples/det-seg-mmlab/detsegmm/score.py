import argparse
import os
from mmdet.apis import init_detector, inference_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--model_folder', default='./model_folder', help='model folder path')
    parser.add_argument('--checkpoint_filename', default='', help='checkpoint filename')
    parser.add_argument('--config_filename', default='', help='config filename')
    parser.add_argument('--test_folder', default='./test_images', help='test image folder')
    parser.add_argument('--prediction_folder', help='output result file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.test_folder):
        os.makedirs(args.test_folder)

    config_file = os.path.join(args.model_folder, args.config_filename)
    checkpoint_file = os.path.join(args.model_folder, args.checkpoint_filename)
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # test a single image and show the results
    imgs = [os.path.join(args.test_folder, img) for img in os.listdir(args.test_folder)]
    print(imgs)
    for i, result in enumerate(inference_detector(model, imgs)):
        show_result(imgs[i], result, model.CLASSES, out_file=os.path.join(args.prediction_folder, 'result_{}.jpg'.format(i)))


if __name__ == '__main__':
    main()
