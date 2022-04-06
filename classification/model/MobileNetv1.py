import warnings
from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

from model.utils._internally_replaced_utils import load_state_dict_from_url
from model.ops.misc import Conv2dNormActivation
from model.utils.utils import _log_api_usage_once
from model.utils._utils import _make_divisible


# from utils._internally_replaced_utils import load_state_dict_from_url
# from ops.misc import Conv2dNormActivation
# from utils.utils import _log_api_usage_once
# from utils._utils import _make_divisible

from torchstat import stat

class Block(nn.Module):
    # Depthwise conv + Pointwise conv
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        layers: List[nn.Module] = []
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    inp,
                    inp,
                    stride=stride,
                    groups=inp,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
                nn.ReLU6(),
            ]
        )
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MobileNetV1(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        cfg_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        
        super().__init__()
        _log_api_usage_once(self)
        
        if block is None:
            block = Block

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        
        input_channel = 32
        last_channel = 1024
          
        if cfg_setting is None:
#             cfg = [(64,1), 
#                    (128,2), 
#                    (128,1), 
#                    (256,2), 
#                    (256,1), 
#                    (512,2), 
#                    (512,1), 
#                    (512,1), 
#                    (512,1), 
#                    (512,1), 
#                    (512,1), 
#                    (1024,2), 
#                    (1024,1)]

            cfg_setting = [
                   [64,1], 
                   [128,2], 
                   [128,1], 
                   [256,2], 
                   [256,1], 
                   [512,2], 
                   [512,1], 
                   [512,1], 
                   [512,1], 
                   [512,1], 
                   [512,1], 
                   [1024,2], 
                   [1024,2],
                  ]

        
        #build first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest) 
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
            
        # build convDW layer   
        for c,s in cfg_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            stride = s
            features.append(block(input_channel, output_channel, stride, expand_ratio = 1, norm_layer=norm_layer))
            input_channel = output_channel
            
            # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def mobilenet_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any)-> MobileNetV1:
    model = MobileNetV1(**kwargs)
    if pretrained:
        arch = "mobilenet_v1"
        
        state_dict = '../pretrain/' + arch + '.pth'
        tdct = torch.load(state_dict)
        tdct.pop('classifier.1.weight')
        tdct.pop('classifier.1.bias')
        model.load_state_dict(tdct, strict=False)
    return model

if __name__=="__main__":
    #Test MobileNet_v1
    model = mobilenet_v1(num_classes = 10)
    
    model_param = sum([param.nelement() for param in model.parameters()])/1e6
    
    print(f'model parameters:{model_param}M')
    
    #input (B,C,H,W)
    input = torch.rand(50,3,32,32)
    output = model(input)
    
    print("input_shape:",input.shape)
    print("output_shape:",output.shape)
    
    stat(model,input.shape[1:])
    
    
            
            
            
            
            
            
        
        
        