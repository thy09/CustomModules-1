import fire
import sys
from torchvision import transforms
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.transformation_directory import ImageTransformationDirectory


class ApplyImageTransform:
    # Follow webservice api contract
    def __init__(self, input_transform_path, meta={}):
        transform_type = meta.get('Transform type', None)
        self.transform = self.get_transforms(input_transform_path, transform_type)
        logger.info(f'Set transform_type {transform_type}, transforms {self.transform}.')
        self.unloader = transforms.ToPILImage()
        logger.info("Transformation init finished.")

    def get_transforms(self, input_transform_path, transform_type):
        loaded_dir = ImageTransformationDirectory.load(input_transform_path)
        if transform_type == 'Train':
            return loaded_dir.torch_transform
        if transform_type == 'Test':
            raw_transforms = loaded_dir.transforms
            test_transforms = [t for t in raw_transforms if not t[0].startswith('Random')]
            return ImageTransformationDirectory.get_torch_transform(test_transforms)
        else:
            # Will never throw this error thanks to UI constraints
            raise TypeError(f"Unsupported transform_type type {transform_type}")

    # Follow webservice api contract
    def apply(self, loaded_dir, meta={}):
        logger.info(f'Applying transform:')
        transformed_dir = loaded_dir.apply_to_images(
            transform=lambda image: self.unloader(
                self.transform(image).squeeze(0)))
        return transformed_dir

    def apply_image_transformation(self, input_image_path, output_path):
        loaded_dir = ImageDirectory.load(input_image_path)
        logger.info("Image dir loaded.")
        transformed_dir = self.apply(loaded_dir)
        transformed_dir.dump(output_path)
        logger.info("Transformed dir dumped")


def entrance(transform_type,
             input_transform_path='../init_transform/',
             input_image_path='../image_dir_test/',
             output_path='../transform_test/'):
    meta = {'Transform type': transform_type}
    task = ApplyImageTransform(input_transform_path, meta)
    task.apply_image_transformation(input_image_path, output_path)


if __name__ == '__main__':
    print("Args=", sys.argv)
    fire.Fire(entrance)
