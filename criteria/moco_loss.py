import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
class MocoLoss(nn.Module):

    def __init__(self):
        super(MocoLoss, self).__init__()
        print("Loading MOCO model from path: {}".format(model_paths["moco"]))
        self.model = self.__load_model()
        self.model.cuda()
        self.model.eval()