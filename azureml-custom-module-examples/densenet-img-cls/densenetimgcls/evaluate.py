from torchvision import datasets
from .utils import get_transform, evaluate, load_model, logger
import os
import fire
import torch


def cal_metric(model, test_set, save_path, batch_size):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    _, _, test_error = evaluate(model=model, loader=test_loader, is_test=True)
    with open(os.path.join(save_path, 'metric.txt'), 'a') as fout:
        fout.write("Top 1 accuracy: {:.5f}\n".format(1 - test_error))
    logger.info('Final test error: {:.4f}, accuracy: {:.4f}'.format(test_error, 1 - test_error))


def entrance(model_path='', model_type='densenet201', memory_efficient=False, num_classes=2,
             data_path='', batch_size=16, save_path=''):
    os.makedirs(save_path, exist_ok=True)
    model = load_model(model_path, model_type, memory_efficient, num_classes=num_classes)
    _, test_transforms = get_transform()
    test_set = datasets.ImageFolder(data_path, transform=test_transforms)
    cal_metric(model, test_set, save_path, batch_size)
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
