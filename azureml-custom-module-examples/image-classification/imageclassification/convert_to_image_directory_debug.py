import fire
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import FolderBasedImageDirectory
from azureml.studio.core.logger import module_host_logger as log, indented_logging_block
from azureml.studio.core.utils.fileutils import iter_files
from azureml.studio.core.logger import TimeProfile
from pathlib import Path


def print_dir_hierarchy_to_log(path):
    log.debug(f"Content of directory {path}:")
    with indented_logging_block():
        for f in iter_files(path):
            log.debug(Path(f).relative_to(path))


def entrance(input_path='../image_dataset/',
             output_path='../image_dir/'):
    logger.info('Start!')
    with TimeProfile(f"Mount/Download dataset to {input_path}"):
        print_dir_hierarchy_to_log(input_path)
    # Case 1: input path is torchvision ImageFolder
    # TODO: Case 2: input path is custom image format
    loader_dir = FolderBasedImageDirectory.load_organized(input_path)
    loader_dir.dump(output_path)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
