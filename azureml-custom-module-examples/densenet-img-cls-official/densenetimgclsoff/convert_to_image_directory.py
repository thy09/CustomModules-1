import fire
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import FolderBasedImageDirectory


def entrance(input_path='/mnt/chjinche/data/small/',
             output_path='/mnt/chjinche/data/image_dir/'):
    logger.info('Start!')
    # Case 1: input path is torchvision ImageFolder
    loader_dir = FolderBasedImageDirectory.load_organized(input_path)
    loader_dir.dump(output_path)
    logger.info('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
