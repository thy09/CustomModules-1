import fire
from azureml.studio.core.logger import logger
from azureml.studio.core.io.transformation_directory import ImageTransformationDirectory


def entrance(resize_size=256,
             center_crop_size=224,
             five_crop=False,
             ten_crop=False,
             pad=False,
             color_jitter=False,
             grayscale=False,
             random_crop=False,
             random_horizontal_flip=True,
             random_vertical_flip=False,
             random_resized_crop=False,
             random_rotation=False,
             random_affine=False,
             random_grayscale=False,
             random_perspective=False,
             random_erasing=False,
             normalize=True,
             output_path='../init_transform/'):
    # Construct image transform
    # TODO: check transforms ordering
    img_trans_dir = ImageTransformationDirectory.create(
        transforms=[('Resize', resize_size), ('CenterCrop', center_crop_size)])
    if five_crop:
        img_trans_dir.append('FiveCrop')
    if ten_crop:
        img_trans_dir.append('TenCrop')
    if pad:
        img_trans_dir.append('Pad')
    if color_jitter:
        img_trans_dir.append('ColorJitter')
    if grayscale:
        img_trans_dir.append('Grayscale')
    if random_crop:
        img_trans_dir.append('RandomCrop')
    if random_horizontal_flip:
        img_trans_dir.append('RandomHorizontalFlip')
    if random_vertical_flip:
        img_trans_dir.append('RandomVerticalFlip')
    if random_resized_crop:
        img_trans_dir.append('RandomResizedCrop')
    if random_rotation:
        img_trans_dir.append('RandomRotation')
    if random_affine:
        img_trans_dir.append('RandomAffine')
    if random_grayscale:
        img_trans_dir.append('RandomGrayscale')
    if random_perspective:
        img_trans_dir.append('RandomPerspective')
    if random_erasing:
        img_trans_dir.append('RandomErasing')
    # Need to do 'ToTensor' op ahead of normalzation.
    img_trans_dir.append('ToTensor')
    if normalize:
        img_trans_dir.append_normalize()
    logger.info(f'Constructed image transforms: {img_trans_dir.transforms}')
    # Dump
    img_trans_dir.dump(output_path)
    logger.info('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
