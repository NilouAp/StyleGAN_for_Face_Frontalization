from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from criteria.lpips.utils import normalize_activation
