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

    @staticmethod
    def __load_model():
        import torchvision.models as models
        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        checkpoint = torch.load(model_paths['moco'], map_location="cpu")
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model