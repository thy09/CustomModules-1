import sys
import torch.nn as nn
import torchvision.models.densenet as densenet
from azureml.studio.core.logger import logger


class DenseNet(nn.Module):
    def __init__(self,
                 model_type='densenet201',
                 pretrained=True,
                 memory_efficient=False,
                 num_classes=20):
        logger.info('Init DenseNet.')
        super(DenseNet, self).__init__()
        if not pretrained:
            model_type = "densenet201"
        densenet_func = getattr(densenet, model_type, None)
        if densenet_func is None:
            # todo: catch exception and throw AttributeError
            logger.info(f"Error: No such pretrained model {model_type}")
            sys.exit()
        logger.info(f"Model type {model_type}, pretrained {pretrained}")
        self.model1 = densenet_func(pretrained=pretrained)
        # Pretrained model is trained based on 1000-class ImageNet dataset
        self.model2 = nn.Linear(1000, num_classes)
        logger.info(f"Model init finished")

    def forward(self, input):
        output = self.model1(input)
        output = self.model2(output)
        return output


if __name__ == '__main__':
    net = DenseNet()
