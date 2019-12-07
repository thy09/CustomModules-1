import sys
import torch.nn as nn
import torchvision.models as models
from azureml.studio.core.logger import logger


class BaseNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs.copy()
        self.pretrained = kwargs.get('pretrained', None)
        # TODO: error if pretrained is None
        self.model_type = kwargs.get('model_type', None)
        logger.info(f'Model config {kwargs}.')
        kwargs.pop('model_type', None)
        logger.info(f'Init {self.model_type}.')
        model_func = getattr(models, self.model_type, None)
        if model_func is None:
            # TODO: catch exception and throw AttributeError
            logger.info(f"Error: No such pretrained model {self.model_type}")
            sys.exit()

        if self.pretrained:
            # Drop 'num_classes' para to avoid size mismatch
            kwargs.pop('num_classes', None)
            logger.info(f'Model config {kwargs}.')
            self.model = model_func(**kwargs)
        else:
            logger.info(f'Model config {kwargs}.')
            self.model = model_func(**kwargs)

        logger.info(f"Model init finished, {self.model}.")

    def forward(self, x):
        out = self.model(x)
        return out
