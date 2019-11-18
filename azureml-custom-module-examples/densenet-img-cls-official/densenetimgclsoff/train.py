# Explicitly load pyarrow.parquet in advance since pyarrow depends on New C++ API on Linux, otherwise segmentation fault
# would occur.
import pyarrow
from torchvision import datasets, transforms
from .densenet import DenseNet
from .utils import (AverageMeter, evaluate, logger, torch_dumper,
                    print_dir_hierarchy_to_log)
from azureml.studio.core.logger import TimeProfile
# from densenet import DenseNet
# from utils import (AverageMeter, get_transform, evaluate, logger,
#                     get_stratified_split_index, torch_dumper,
#                     print_dir_hierarchy_to_log)
import time
import fire
import torch
from azureml.studio.core.io.model_directory import save_model_to_directory


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
          model_config,
          idx_to_class_dict,
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
    logger.info('torch setting')
    # torch cuda random seed setting
    if random_seed is not None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(random_seed)
            else:
                torch.cuda.manual_seed(random_seed)
        else:
            torch.manual_seed(random_seed)
    logger.info("Data start loading")
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
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)
    logger.info('Start training epochs')
    best_error = 1
    counter = 0
    best_checkpoint_name = 'best_model.pth'
    config_file_name = 'model_config.json'
    id_to_class_file_name = 'id_to_class.json'
    for epoch in range(epochs):
        # scheduler.step()
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

        logger.info(
            f'valid loss did not decrease consecutively for {counter} epoch')
        # todo: save checkpoint files, but removed now to increase web service deployment efficiency
        # checkpoint_path = os.path.join(save_model_path, 'model_epoch_{}.pth'.format(state["epoch"]))
        # torch.save(state["state_dict"], checkpoint_path)
        logger.info(','.join([
            f'Epoch {epoch + 1:d}', f'train_loss {train_loss:.6f}',
            f'train_error {train_error:.6f}', f'valid_loss {valid_loss:.5f}',
            f'valid_error {valid_error:.5f}'
        ]))

        if is_best:
            logger.info(
                f'Get better top1 accuracy: {1-best_error:.4f} saving weights to {best_checkpoint_name}'
            )
            dumper = torch_dumper(state_dict, model_config, idx_to_class_dict,
                                  best_checkpoint_name, config_file_name,
                                  id_to_class_file_name)
            save_model_to_directory(save_model_path, dumper)
            # shutil.copyfile(checkpoint_path, best_checkpoint_path)

        early_stop = True if counter >= patience else False
        if early_stop:
            logger.info("early stopped.")
            break


def entrance(train_data_path='/mnt/chjinche/data/output_transformed/',
             valid_data_path='/mnt/chjinche/data/output_transformed/',
             save_model_path='/mnt/chjinche/projects/saved_model',
             model_type='densenet201',
             pretrained=True,
             memory_efficient=False,
             epochs=1,
             batch_size=16,
             learning_rate=0.001,
             random_seed=231,
             patience=2):
    logger.info("Start training.")
    to_tensor_transform = transforms.Compose([transforms.ToTensor()])
    # logger.info(f"data path: {train_data_path}")
    with TimeProfile(f"Mount/Download dataset to '{train_data_path}'"):
        print_dir_hierarchy_to_log(train_data_path)
    with TimeProfile(f"Mount/Download dataset to '{valid_data_path}'"):
        print_dir_hierarchy_to_log(valid_data_path)
    # No RandomHorizontalFlip in validation
    train_set = datasets.ImageFolder(train_data_path,
                                     transform=to_tensor_transform)
    print(train_set.classes)
    valid_set = datasets.ImageFolder(valid_data_path,
                                     transform=to_tensor_transform)
    # assert the same classes between train_set and valid_set
    logger.info("Made dataset")
    classes = train_set.classes
    num_classes = len(classes)
    idx_to_class_dict = {i: classes[i] for i in range(num_classes)}
    logger.info("Start constructing model")
    model = DenseNet(model_type=model_type,
                     pretrained=pretrained,
                     memory_efficient=memory_efficient,
                     num_classes=num_classes)
    model_config = {
        'model_type': model_type,
        'pretrained': pretrained,
        'memory_efficient': memory_efficient,
        'num_classes': num_classes
    }
    train(model=model,
          model_config=model_config,
          idx_to_class_dict=idx_to_class_dict,
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
