# from sklearn.model_selection import train_test_split
# from .densenet import DenseNet
from .densenet import DenseNet
import torch
import os
import time
import json
from azureml.studio.core.logger import logger
from azureml.studio.core.utils.fileutils import ensure_folder, iter_files
from azureml.studio.core.logger import module_host_logger as log, indented_logging_block
from pathlib import Path


_IDENTIFIER_NAME = 'identifier'
_LABEL_NAME = 'label'


class ScoreColumnConstants:
    # Label and Task Type Region
    BinaryClassScoredLabelType = "Binary Class Assigned Labels"
    MultiClassScoredLabelType = "Multi Class Assigned Labels"
    RegressionScoredLabelType = "Regression Assigned Labels"
    ClusterScoredLabelType = "Cluster Assigned Labels"
    ScoredLabelsColumnName = "Scored Labels"
    ClusterAssignmentsColumnName = "Assignments"
    # Probability Region
    CalibratedScoreType = "Calibrated Score"
    ScoredProbabilitiesColumnName = "Scored Probabilities"
    ScoredProbabilitiesMulticlassColumnNamePattern = "Scored Probabilities"
    # Distance Region
    ClusterDistanceMetricsColumnNamePattern = "DistancesToClusterCenter no."


def _filter_column_names_with_prefix(name_list, prefix=''):
    # if prefix is '', all string.startswith(prefix) is True.
    if prefix == '':
        return name_list
    return [column_name for column_name in name_list if column_name.startswith(prefix)]


def generate_score_column_meta(predict_df):
    score_columns = {x: x for x in _filter_column_names_with_prefix(
        predict_df.columns.tolist(), prefix=ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern)}
    score_columns[ScoreColumnConstants.MultiClassScoredLabelType] = ScoreColumnConstants.ScoredLabelsColumnName
    logger.info("Multi-class Classification Model Scored Columns are: ")
    return score_columns


def torch_loader(load_from_dir, model_spec):
    """Load the pickle model by reading the file indicated by file_name in model_spec."""
    model_file_name = model_spec['model_file_name']
    config_file_name = model_spec['config_file_name']
    id_to_class_file_name = model_spec['id_to_class_file_name']
    model_config = json.load(
        open(os.path.join(load_from_dir, config_file_name)))
    id_to_class_dict = json.load(
        open(os.path.join(load_from_dir, id_to_class_file_name)))
    model = DenseNet(model_type=model_config['model_type'],
                     pretrained=False,
                     memory_efficient=model_config['memory_efficient'],
                     num_classes=model_config['num_classes'])
    model.load_state_dict(
        torch.load(os.path.join(load_from_dir, model_file_name),
                   map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
    return model, id_to_class_dict


def torch_dumper(state_dict,
                 model_config,
                 id_to_class_dict,
                 model_file_name=None,
                 config_file_name=None,
                 id_to_class_file_name=None):
    """Return a dumper to save torch state dict."""
    if not model_file_name:
        model_file_name = '_model.pth'

    if not config_file_name:
        config_file_name = '_config.json'

    if not id_to_class_file_name:
        id_to_class_file_name = '_id_to_class.json'

    def model_dumper(save_to):
        model_full_path = os.path.join(save_to, model_file_name)
        config_full_path = os.path.join(save_to, config_file_name)
        id_to_class_full_path = os.path.join(save_to, id_to_class_file_name)
        ensure_folder(os.path.dirname(os.path.abspath(model_full_path)))
        ensure_folder(os.path.dirname(os.path.abspath(config_full_path)))
        ensure_folder(os.path.dirname(os.path.abspath(id_to_class_full_path)))
        torch.save(state_dict, model_full_path)
        json.dump(model_config, open(config_full_path, "w"))
        json.dump(id_to_class_dict, open(id_to_class_full_path, "w"))
        model_spec = {
            'model_type': 'torch_state_dict',
            'model_file_name': model_file_name,
            'config_file_name': config_file_name,
            'id_to_class_file_name': id_to_class_file_name,
        }
        return model_spec

    return model_dumper


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate(model, loader, print_freq=1, is_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on eval mode
    model.eval()
    end = time.time()
    target_list = []
    pred_top1_list = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create variables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            target_list += target.tolist()
            pred_top1_list += [i[0] for i in pred.tolist()]
            error.update(
                torch.ne(pred.squeeze(), target.cpu()).float().sum().item() /
                batch_size, batch_size)
            losses.update(loss.item() / batch_size, batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(loader)),
                    'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(
                        batch_time.val, batch_time.avg),
                    'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(
                        losses.val, losses.avg),
                    'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(
                        error.val, error.avg),
                ])
                logger.info(res)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, (target_list, pred_top1_list)


# def get_stratified_split_index(n_samples, class_idx_list, valid_size):
#     sample_ids = np.array(list(range(n_samples)))
#     labels = np.array(class_idx_list)
#     train_index, valid_index, train_label, valid_label = train_test_split(
#         sample_ids, labels, stratify=labels, test_size=valid_size)
#     return train_index, valid_index


def print_dir_hierarchy_to_log(path):
    log.debug(f"Content of directory {path}:")
    with indented_logging_block():
        for f in iter_files(path):
            log.debug(Path(f).relative_to(path))
