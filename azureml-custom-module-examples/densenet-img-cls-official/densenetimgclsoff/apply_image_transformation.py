import json
import fire
from pathlib import Path
from torchvision import transforms
# from .utils import get_transform, load_model, logger
from azureml.studio.core.logger import TimeProfile
from .utils import logger, print_dir_hierarchy_to_log
from azureml.studio.core.io.image_directory import ImageDirectory

# mean and stdv of imagenet dataset.
# Usually if you use case in the same data domain as imagenet,
# the mean and std won’t be that different and you can try to use the ImageNet statistics.
# todo: If you are dealing with another domain, e.g. medical images,
# re-calculate stats on your own data is recommended.
MEAN = [0.485, 0.456, 0.406]
STDV = [0.229, 0.224, 0.225]


class ApplyImageTransform:
    def __init__(self, input_transform_path):
        self.transform = self.get_transform(input_transform_path)
        self.unloader = transforms.ToPILImage()

    def get_transform(self, input_transform_path):
        with open(Path(input_transform_path) / 'transform.json', 'r') as f:
            transform = json.load(f)

        resize_size = transform.get('resize_size', None)
        center_crop_size = transform.get('center_crop_size', None)
        random_horizontal_flip = transform.get('random_horizontal_flip', None)
        normalize = transform.get('normalize', None)
        # todo check whether above paras are None
        if random_horizontal_flip is True:
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(center_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STDV, inplace=True),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STDV),
            ])
        return transform

    def apply(self, loaded_dir):
        print(self.transform)
        transformed_dir = loaded_dir.apply_to_images(
            transform=lambda image: self.unloader(
                self.transform(image).squeeze(0)))

        return transformed_dir

    def apply_image_transformation(self, input_image_path, output_path):
        Path(output_path).mkdir(exist_ok=True)
        loaded_dir = ImageDirectory.load(input_image_path)
        transformed_dir = self.apply(loaded_dir)
        transformed_dir.dump(output_path)
        logger.info("Transformed dir dumped")


def entrance(input_transform_path='/mnt/chjinche/transform/train/',
             input_image_path='/mnt/chjinche/data/small/',
             output_path='/mnt/chjinche/data/output_transformed/'):
    with TimeProfile(f"Mount/Download dataset to '{input_image_path}'"):
        print_dir_hierarchy_to_log(input_image_path)

    task = ApplyImageTransform(input_transform_path)
    task.apply_image_transformation(input_image_path, output_path)


if __name__ == '__main__':
    fire.Fire(entrance)
