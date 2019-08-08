import argparse
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder",
                        default='/home/root/chjinche/projects/balloon/train',
                        type=str,
                        help="The input dataset folder.")
    parser.add_argument("--out_dataset_folder",
                        default="",
                        type=str,
                        help="Output dataset pickle folder.")
    parser.add_argument("--out_dataset_file",
                        default="",
                        type=str,
                        help="Output dataset pickle file.")
    return parser


def train_opts():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name',
    #                     type=str,
    #                     required=True,
    #                     help='custom dataset name')
    # parser.add_argument('--num_classes',
    #                     type=int,
    #                     required=True,
    #                     help='number of classes for custom dataset')
    parser.add_argument('--pretrained_model_folder',
                        default='./pretrained_model_folder',
                        help='pretrained model folder')
    parser.add_argument('--pretrained_model_file',
                        default='mask_rcnn_coco.h5',
                        help='pretrained_model_file')
    parser.add_argument('--dataset_train_folder',
                        default='./out_train',
                        help='training dataset folder')
    parser.add_argument('--dataset_train_file',
                        default='train.pkl',
                        help='training dataset file')
    parser.add_argument('--dataset_val_folder',
                        default='./out_val',
                        help='validation dataset folder')
    parser.add_argument('--dataset_val_file',
                        default='val.pkl',
                        help='validation dataset file')
    parser.add_argument('--model_folder',
                        default='./balloon_logs',
                        help='logs and checkpoints directory')
    parser.add_argument('--gpu_cnt',
                        default=2,
                        type=int,
                        help='gpu count')
    parser.add_argument('--step_per_epoch',
                        default=100,
                        type=int,
                        help='number of training steps per epoch')
    parser.add_argument('--det_min_conf',
                        default=0.9,
                        type=float,
                        help='confidence of detection skip')
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        help='number of training epochs')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate')
    return parser


def score_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder',
                        default='./logs',
                        help='logs and checkpoints folder')
    parser.add_argument('--prediction_folder',
                        default='./pred',
                        help='prediction folder')
    parser.add_argument('--dataset_score_folder',
                        default='./out_val',
                        help='testing dataset folder')
    parser.add_argument('--dataset_score_file',
                        default='val.pkl',
                        help='testing dataset file')
    return parser
