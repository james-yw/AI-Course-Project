import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


from model.utils._internally_replaced_utils import load_state_dict_from_url
from model.ops.misc import Conv2dNormActivation
from model.utils.utils import _log_api_usage_once
from model.utils._utils import _make_divisible

# from utils._internally_replaced_utils import load_state_dict_from_url
# from ops.misc import Conv2dNormActivation
# from utils.utils import _log_api_usage_once
# from utils._utils import _make_divisible
# from torchstat import stat
__all__ = ["ConvNet", "convnet"]

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d\
            (in_planes, out_planes, kernel_size=3, stride=stride, 
             padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
class ConvNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, 
    # by default conv stride=1
    #Mobilenet是一个Conv和若干个Depthwise conv和全连接的组合
    #64指第一个Depthwise conv模块输出特征图通道数为64
    #(128,2)指第二个Depthwise conv模块输出特征图通道数为128但以stride为2进行卷积，目的是倍减图大小，以此代替pooling
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 
           512, 512, 512, 512, 512, 
           (1024,2), (1024,2)]

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, 
        	stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def convnet(progress: bool = True, **kwargs: Any) -> ConvNet:
    model = ConvNet(**kwargs)
    
    return model

if __name__=="__main__":
    #Test MobileNet_v1
    
    model = convnet(num_classes=10)
    
    model_param = sum([param.nelement() for param in model.parameters()])/1e6
    
    print(f'model parameters:{model_param}M')
    
    #input (B,C,H,W)
    input = torch.rand(50,3,32,32)
    output = model(input)
    
    print("input_shape:",input.shape)
    print("output_shape:",output.shape)
    
    stat(model,input.shape[1:])