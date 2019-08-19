from .utils import logger
from .mrcnn import utils
import cloudpickle
import os
import json
import numpy as np
import skimage.draw
import fire


class CustomDataset(utils.Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        self.source_name = "custom"
        self.input_class_name_list = []
        self.num_classes = len(self.input_class_name_list)

    def set_attr(self, input_class_name_list):
        self.input_class_name_list = input_class_name_list

    def load_custom_dataset(self, dataset_dir):
        """Load a subset of the Balloon dataset."""
        if not os.path.exists(dataset_dir):
            logger.info("dataset folder does not exist.")
            return
        # Add classes. We have only one class to add.
        # add_class(self, source, class_id, class_name)
        for i, cls_name in enumerate(self.input_class_name_list):
            self.add_class(self.source_name, i+1, cls_name[i])
        # Train or validation dataset?
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                source=self.source_name,
                image_id=a['filename'],  # use file name as a unique image id
                path='',
                image=image,
                width=width, height=height,
                polygons=polygons)

    def add_image(self, source, image_id, path, image, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "image": image
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # image = skimage.io.imread(self.image_info[image_id]['path'])
        image = self.image_info[image_id]['image']
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if (image_info["source"] != self.source_name) or (image_info["polygons"] is None):
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


def test(dataset_folder, out_dataset_folder):
    os.makedirs(out_dataset_folder, exist_ok=True)
    input_cls_name_path = os.path.join(dataset_folder, "input_cls_name.json")
    with open(input_cls_name_path, 'r') as fin:
        input_cls_names = json.load(fin)["input_cls_names"]
    dataset = CustomDataset()
    dataset.set_attr(input_cls_names)
    dataset.load_custom_dataset(dataset_folder)
    dataset.prepare()
    # Output
    out_file_path = os.path.join(out_dataset_folder, "dataset.pkl")
    with open(out_file_path, 'wb') as fout:
        cloudpickle.dump(dataset, fout)
    output_mapping_file_path = os.path.join(out_dataset_folder, "mapping.json")
    with open(output_mapping_file_path, 'w') as fin:
        json.dump(dataset.class_names, fin)


if __name__ == '__main__':
    fire.Fire(test)
