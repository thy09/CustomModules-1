import fire
from azureml.studio.core.logger import logger, indented_logging_block
from azureml.studio.core.utils.fileutils import iter_files
from azureml.studio.core.logger import TimeProfile
from pathlib import Path


def print_dir_hierarchy_to_log(path):
    logger.debug(f"Content of directory {path}:")
    with indented_logging_block():
        for f in iter_files(path):
            logger.debug(Path(f).relative_to(path))


def entrance(input_path='../image_dataset/',
             output_path='../image_dir/'):
    logger.info('Start!')
    with TimeProfile(f"Mount/Download dataset to {input_path}"):
        print_dir_hierarchy_to_log(input_path)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
