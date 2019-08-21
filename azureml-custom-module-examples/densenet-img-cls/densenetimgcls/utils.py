from torchvision import datasets, transforms
from .densenet import MyDenseNet
import shutil
import torch
import os
import sys
import time


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


def save_checkpoint(state, is_best, save_path, patience):
    checkpoint_path = os.path.join(save_path, 'model_epoch_{}.pth'.format(state["epoch"]))
    torch.save(state["state_dict"], checkpoint_path)
    log_path = os.path.join(save_path, 'log.txt')
    with open(log_path, 'a') as fout:
        fout.write('Epoch {:3d},{:.6f}, train_error {:.6f}, valid_loss {:.5f}, valid_error {:.5f}\n'
                   .format(state["epoch"], state["train_loss"], state["train_error"],
                           state["valid_loss"], state["valid_error"]))
    if is_best:
        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
        message = "Get better top1 accuracy: {:.4f} saving weights to {}\n".format(state["best_accuracy"], best_checkpoint_path)
        print(message)
        with open(os.path.join(save_path, 'log.txt'), 'a') as fout:
            fout.write(message)
        shutil.copyfile(checkpoint_path, best_checkpoint_path)
    early_stop = True if state["counter"] >= patience else False
    if early_stop:
        print("early stopped.")
        with open(os.path.join(save_path, 'log.txt'), 'a') as fout:
            fout.write("early stopped.\n")
        sys.exit()


def get_transform():
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    return train_transforms, test_transforms


def load_model(model_path, model_type, memory_efficient, num_classes):
    model = MyDenseNet(model_type=model_type, pretrained=False, memory_efficient=memory_efficient, classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth'), map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
    return model


def evaluate(model, loader, print_freq=1, is_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on eval mode
    model.eval()
    end = time.time()
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
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [{:d}/{:d}]'.format(batch_idx + 1, len(loader)),
                    'Avg_Time_Batch/Avg_Time_Epoch: {:.3f}/{:.3f}'.format(batch_time.val, batch_time.avg),
                    'Avg_Loss_Batch/Avg_Loss_Epoch: {:.4f}/{:.4f}'.format(losses.val, losses.avg),
                    'Avg_Error_Batch/Avg_Error_Epoch: {:.4f}/{:.4f}'.format(error.val, error.avg),
                ])
                print(res)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg
