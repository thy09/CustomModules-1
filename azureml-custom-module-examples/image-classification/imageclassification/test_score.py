from azureml.designer.model.io import load_generic_model
from azureml.studio.core.io.image_directory import ImageDirectory
import torch
import torch.nn as nn
from torchvision import transforms
# from .densenet import DenseNet
from .utils import logger


if __name__ == '__main__':
    # Test inference
    print("Testing inference.")
    loaded_generic_model = load_generic_model(
        path='/mnt/chjinche/projects/saved_model')
    model = loaded_generic_model.raw_model
    # # check predict before save
    # state_dict = model.state_dict()
    loader_dir = ImageDirectory.load('/mnt/chjinche/data/out_transform_test/')
    # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
    # model_config = {
    #     'model_type': 'densenet201',
    #     'pretrained': False,
    #     'memory_efficient': True,
    #     'num_classes': 3
    # }
    # new_model = DenseNet(**model_config)
    # new_model.load_state_dict(state_dict)
    # if torch.cuda.is_available():
    #     new_model = new_model.cuda()
    #     if torch.cuda.device_count() > 1:
    #         new_model = torch.nn.DataParallel(new_model).cuda()
    # new_model.eval()
    # for img, label, identifier in loader_dir.iter_images():
    #     # Convert PIL image to tensor
    #     input_tensor = to_tensor_transform(img)
    #     input_tensor = input_tensor.unsqueeze(0)
    #     if torch.cuda.is_available():
    #         input_tensor = input_tensor.cuda()
    #     with torch.no_grad():
    #         output = new_model(input_tensor)
    #         softmax = nn.Softmax(dim=1)
    #         pred_probs = softmax(output).cpu().numpy()[0]
    #         logger.info(f'pred_probs {pred_probs}')
    # loader_dir = ImageDirectory.load('/mnt/chjinche/data/output_transformed_test/')
    # result_dfd = loaded_generic_model.predict(loader_dir.iter_images())
    result_dfd = model(loader_dir.iter_images())
    print(f'result_dfd: {result_dfd}')
