# Explicitly load pyarrow.parquet in advance since pyarrow depends on New C++ API on Linux, otherwise segmentation fault
# would occur.
import pyarrow
from torchvision import datasets
from .densenet import DenseNet
from .utils import save_checkpoint, AverageMeter, get_transform, evaluate, logger, get_stratified_split_index
# from densenet import DenseNet
# from utils import save_checkpoint, AverageMeter, get_transform, evaluate, logger, get_stratified_split_index
import os
import time
import fire
import torch
import pandas as pd
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory


def train_epoch(model, loader, optimizer, epoch, epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on train mode
    model.train()
    end = time.time()
    batches = len(loader)
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
        error.update(
            torch.ne(pred.squeeze(), target.cpu()).float().sum().item() /
            batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                f'Epoch: [{epoch + 1}/{epochs}]',
                f'Iter: [{batch_idx + 1}/{batches}]',
                f'Avg_Time_Batch/Avg_Time_Epoch: {batch_time.val:.3f}/{batch_time.avg:.3f}',
                f'Avg_Loss_Batch/Avg_Loss_Epoch: {losses.val:.4f}/{losses.avg:.4f}',
                f'Avg_Error_Batch/Avg_Error_Epoch: {error.val:.4f}/{error.avg:.4f}'
            ])
            logger.info(res)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model,
          train_set,
          valid_set,
          save_model_path,
          epochs,
          batch_size,
          lr=0.001,
          wd=0.0001,
          momentum=0.9,
          random_seed=None,
          patience=10):
    # torch cuda random seed setting
    if random_seed is not None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(random_seed)
            else:
                torch.cuda.manual_seed(random_seed)
        else:
            torch.manual_seed(random_seed)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(torch.cuda.is_available()),
        num_workers=0)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        num_workers=0)
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)
    logger.info('Start training')
    best_error = 1
    counter = 0
    for epoch in range(epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(model=model,
                                                 loader=train_loader,
                                                 optimizer=optimizer,
                                                 epoch=epoch,
                                                 epochs=epochs)
        _, valid_loss, valid_error, _ = evaluate(model=model,
                                                 loader=valid_loader)
        # Determine if model is the best
        if valid_error < best_error:
            is_best = True
            best_error = valid_error
        else:
            is_best = False
        state_dict = model.module.state_dict(
        ) if torch.cuda.device_count() > 1 else model.state_dict()
        # early stop
        if epoch == 0:
            last_epoch_valid_loss = valid_loss
        else:
            if valid_loss >= last_epoch_valid_loss:
                counter += 1
            else:
                counter = 0
            last_epoch_valid_loss = valid_loss
        logger.info(f'valid loss did not decrease consecutively for {counter} epoch')
        early_stop = save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                "best_error": best_error,
                "best_accuracy": 1 - best_error,
                "train_loss": train_loss,
                "train_error": train_error,
                "valid_loss": valid_loss,
                "valid_error": valid_error,
                "counter": counter
            }, is_best, save_model_path, patience)
        if early_stop:
            break


def entrance(data_path='/mnt/chjinche/data/small/',
             save_model_path='/mnt/chjinche/projects/saved_model',
             save_classes_path='/mnt/chjinche/projects/saved_classes',
             model_type='densenet201',
             pretrained=True,
             memory_efficient=False,
             epochs=1,
             batch_size=16,
             learning_rate=0.001,
             random_seed=231,
             patience=2):
    train_transforms, test_transforms = get_transform()
    # No RandomHorizontalFlip in validation
    train_set = datasets.ImageFolder(data_path, transform=train_transforms)
    valid_set = datasets.ImageFolder(data_path, transform=test_transforms)
    classes_df = pd.DataFrame({'classes': train_set.classes})
    save_data_frame_to_directory(save_classes_path, data=classes_df)
    class_idx_list = [sample[1] for sample in train_set.samples]
    train_index, valid_index = get_stratified_split_index(
        n_samples=len(train_set), class_idx_list=class_idx_list, valid_size=0.1)
    train_set = torch.utils.data.Subset(train_set, train_index)
    valid_set = torch.utils.data.Subset(valid_set, valid_index)
    model = DenseNet(model_type=model_type,
                     pretrained=pretrained,
                     memory_efficient=memory_efficient,
                     classes=classes_df.shape[0])
    os.makedirs(save_model_path, exist_ok=True)
    train(model=model,
          train_set=train_set,
          valid_set=valid_set,
          save_model_path=save_model_path,
          epochs=epochs,
          batch_size=batch_size,
          lr=learning_rate,
          random_seed=random_seed,
          patience=patience)
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
