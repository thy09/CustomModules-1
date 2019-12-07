import fire
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import FolderBasedImageDirectory


def entrance(input_path='../image_dataset/',
             output_path='../image_dir/'):
    logger.info('Start!')
    # Case 1: input path is torchvision ImageFolder
    # TODO: Case 2: input path is custom image format
    loader_dir = FolderBasedImageDirectory.load_organized(input_path)
    loader_dir.dump(output_path)
    logger.info('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
