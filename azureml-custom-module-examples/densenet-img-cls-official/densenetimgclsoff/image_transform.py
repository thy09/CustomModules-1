import json
import fire
import os
from pathlib import Path
from azureml.studio.core.io.any_directory import AnyDirectory


def entrance(resize_size=256,
             center_crop_size=224,
             random_horizontal_flip=True,
             normalize=True,
             output_train_transform_path='/mnt/chjinche/transform/train/',
             output_test_transform_path='/mnt/chjinche/transform/test/'):
    os.makedirs(output_train_transform_path, exist_ok=True)
    os.makedirs(output_test_transform_path, exist_ok=True)
    transform = {
        'resize_size': resize_size,
        'center_crop_size': center_crop_size,
        'random_horizontal_flip': random_horizontal_flip,
        'normalize': normalize
    }
    print(transform)
    with open(Path(output_train_transform_path)/'transform.json', 'w') as f:
        json.dump(transform, f)
    transform.pop('random_horizontal_flip', None)
    with open(Path(output_test_transform_path)/'transform.json', 'w') as f:
        json.dump(transform, f)
    # workaround with anydirectory dump
    any_dir = AnyDirectory(meta={'type': 'TransformationDirectory'})
    any_dir.dump(save_to=output_train_transform_path)
    any_dir.dump(save_to=output_test_transform_path)


if __name__ == '__main__':
    fire.Fire(entrance)
