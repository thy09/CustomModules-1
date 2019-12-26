import fire
from pathlib import Path
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory
# from azureml.studio.core.io.image_directory import FolderBasedImageDirectory


def entrance(input_path='/mnt/chjinche/test_data/',
             output_path='/mnt/chjinche/test_data/image_dir/'):
    logger.info('Start!')
    # Case 1: input path is torchvision ImageFolder
    # loader_dir = FolderBasedImageDirectory.load_organized(input_path)
    # TODO: Case 2: input path is custom image format
    compressed_extensions = {'.tar', '.zip'}
    compressed_path = None
    for path in Path(input_path).glob(r'**/*'):
        if path.suffix in compressed_extensions:
            compressed_path = path

    logger.info(f'compressed file path {compressed_path}')
    loader_dir = ImageDirectory.load_compressed(compressed_path)
    loader_dir.dump(output_path)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
