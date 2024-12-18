import torch
import torch.nn as nn

from criteria.lpips.networks import get_network, LinLayers
from criteria.lpips.utils import get_state_dict


class LPIPS(nn.Module):

    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
