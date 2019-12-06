import fire
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory, FolderBasedImageDirectory
from .utils import get_stratified_split_list


def split_images(src_path, tgt_train_path, tgt_test_path, fraction):
    loaded_dir = ImageDirectory.load(src_path)
    lst = loaded_dir.image_lst
    logger.info(f'Start splitting.')
    train_set_lst, test_set_lst = get_stratified_split_list(lst, fraction)
    logger.info(f'Got stratified split list. train {len(train_set_lst)}, test {len(test_set_lst)}.')
    train_set_dir = FolderBasedImageDirectory.create_with_lst(src_path, train_set_lst)
    test_set_dir = FolderBasedImageDirectory.create_with_lst(src_path, test_set_lst)
    logger.info('Dump train set.')
    train_set_dir.dump(tgt_train_path)
    logger.info('Dump test set.')
    test_set_dir.dump(tgt_test_path)


def entrance(src_path='/mnt/chjinche/data/image_dir/',
             fraction=0.9,
             tgt_train_path='/mnt/chjinche/data/image_dir_train/',
             tgt_test_path='/mnt/chjinche/data/image_dir_test/'):
    logger.info('Start!')
    split_images(src_path, tgt_train_path, tgt_test_path, fraction)
    logger.info('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
