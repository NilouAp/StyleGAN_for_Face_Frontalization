from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from criteria.lpips.utils import normalize_activation

def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')
