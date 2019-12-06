import torch
import time
from azureml.studio.core.logger import logger


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
