import argparse
import os
from .mmdet.apis import init_detector, inference_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--model_folder', default='./model_folder', help='model folder path')
    parser.add_argument('--checkpoint_filename', default='', help='checkpoint filename')
    parser.add_argument('--config_filename', default='', help='config filename')
    parser.add_argument('--test_folder', default='./test_images', help='test image folder')
    parser.add_argument('--prediction_folder', help='output result file')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
    #     help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.test_folder):
        os.makedirs(args.test_folder)

    # config_file = os.path.join(args.model_folder, 'faster_rcnn_r50_fpn_1x.py')
    # checkpoint_file = os.path.join(args.model_folder, 'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
    config_file = os.path.join(args.model_folder, args.config_filename)
    checkpoint_file = os.path.join(args.model_folder, args.checkpoint_filename)
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # test a single image and show the results
#    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    imgs = os.listdir(args.test_folder)
    imgs = [os.path.join(args.test_folder, img) for img in imgs]
    print(imgs)
    # result = inference_detector(model, img)
    # show_result(img, result, model.CLASSES, out_file='./test_result.jpg')
    for i, result in enumerate(inference_detector(model, imgs)):
        show_result(imgs[i], result, model.CLASSES, out_file=os.path.join(args.prediction_folder, 'result_{}.jpg'.format(i)))


if __name__ == '__main__':
    main()
