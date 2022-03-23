import os
import copy
import torch
import torch.nn as nn
import numpy as np
from model.alexnet import alexnet
from model.MobileNetv1 import mobilenet_v1
from model.MobileNetv2 import mobilenet_v2
from model.MobileNetv3 import mobilenet_v3_large, mobilenet_v3_small
from model.squeezenet import squeezenet1_0

from torchstat import stat

torch.backends.cudnn.enabled = False

class NetModel(object):
    """ Class for Network Model """
    def __init__(self, config):
        
        self.train_cpu = config.cpu
        self.gpu = config.gpu
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.num_classes = config.num_classes
        self.checkpoint_path = config.checkpoint_path
        self.network_name = config.net.replace(" ", "_").lower()
        
        self.net = config.net
        self.opt = config.optimizer
        self.pretrain = config.pretrain
        self.pretrained_model_path = config.pretrained_model_path
        self.drop_layer = config.drop_layer
        
        self.loss = config.loss
        
        self._set_net_model()
        
       
    def _set_net_model(self):
        if torch.cuda.is_available() and not self.train_cpu:
            self.device = torch.device("cuda:{}".format(self.gpu))
            print(f"Training On GPU, device{self.gpu}...")
        else:
            self.device = torch.device("cpu")
            print(f"Training On CPU...")
            
        self.network = self._get_network().to(self.device)
        print(self.network)
        
        #loss function
        
        if self.loss == 'CrossEntropyLoss': 
            self.criterion = nn.CrossEntropyLoss()
        
        elif self.loss == 'Focal_loss':
            self.criterion = Focal_loss()
            
        
        # Optimizer
        if self.opt == 'Adadelta' or 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.network.parameters(), rho=0.95, eps=1e-8, lr=self.learning_rate)
        elif self.opt == 'SGD' or 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0)
        elif self.opt == 'Adam' or 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.opt == 'AdaGrad' or 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.network.parameters(), lr=self.learning_rate, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        elif self.opt == 'RMSProp' or 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        else:
            raise ValueError(f"Unsupport {self.optimizer}, Please check out the variables settings!")
            
            
    
    def _get_network(self):
        network = Classifier(self.net, self.device, self.num_classes, self.pretrain, self.pretrained_model_path, self.drop_layer)
        return network
    
    def train(self, data, label, to_softmax=True):
        # set network status
        self.network.train()
        self.network.zero_grad()
        # convert data and label to gpu
        _data = data.to(self.device)
        _label = label.to(self.device)
        _network_output = self.network(_data)
        # compute error and bp
        
        err = self.criterion(_network_output, _label)
        err.backward()
        self.optimizer.step()
        
        if to_softmax:
            return err.cpu().item(), torch.softmax(_network_output.detach(), dim=1)
        else:
            return err.cpu().item(), _network_output.detach()
        
    def predict(self, data, label, to_softmax=True):
        # set network status
        self.network.eval()
        # convert data and label to gpu
        _data = data.to(self.device)
        _label = label.to(self.device)
        _network_output = self.network(_data)
        err = self.criterion(_network_output, _label)
        if to_softmax:
            return err.cpu().item(), torch.softmax(_network_output.detach(), dim=1)
        else:
            return err.cpu().item(), _network_output.detach()
    
    def inference(self, data, to_softmax=True):
        # set network status
        self.network.eval()
        # convert data and label to gpu
        _data = data.to(self.device)
        _network_output = self.network(_data)
        if to_softmax:
            return torch.exp(_network_output.detach())
        else:
            return _network_output.detach()
    
    def is_correct(self, label, output):
        label = label.to(self.device)
        predicts = torch.argmax(output, dim=1)
        count = torch.sum(predicts==label)
        count = count.cpu().item()
        return count
        
    def save_params(self, epoch, checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = "{}/{}_epoch_{}.pth".format(checkpoint_path, self.network_name, epoch)
        torch.save(self.network.state_dict(), save_path)
        print("Save network parameters to {}".format(save_path))
        
    def load_params(self, epoch=0, checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
            load_path = "{}/{}_epoch_{}.pth".format(checkpoint_path, self.network_name, epoch)
        else:
            load_path = checkpoint_path
        self.network.load_state_dict(torch.load(load_path, map_location=self.device))
        print("Load network parameters from {}".format(load_path))
     
    def model_param(self, data):
        
        model = Classifier(self.net, self.num_classes)
        model_param = sum([param.nelement() for param in model.parameters()])/1e6
        print(f'model parameters:{model_param}M')
        stat(model,data.shape[1:])
        
class Classifier(nn.Module):
    def __init__(self, net, device, num_classes, pretrain = False, pretrained_model_path = None, drop_layer = None):
        super(Classifier, self).__init__()
        
        if net == 'AlexNet':
            m = alexnet(num_classes=10)
        elif net == 'mobilenet_v1':
            m = mobilenet_v1(num_classes=10)  
        elif net == 'mobilenet_v1':
            m = mobilenet_v1(num_classes=10)
        elif net == 'mobilenet_v2':
            m = mobilenet_v2(num_classes=10)
        elif net == 'mobilenet_v3_small':
            m = mobilenet_v3_small(num_classes=10)
        elif net == 'mobilenet_v3_large':
            m = mobilenet_v3_large(num_classes=10) 
        else:
            raise ValueError(f"Unsupport {self.net}, Please check out the variables settings!")
        
        if pretrain:
            tdct = self.load_model_param(pretrained_model_path, drop_layer, device)
            m.load_state_dict(tdct, strict=False)
            print("Pre-trained Model Parameters:", tdct.keys())

        model_param = sum([param.nelement() for param in m.parameters()])/1e6
        print(f'model parameters:{model_param}M')
   
        self.features = m
           
    def load_model_param(self, pretrained_model_path, drop_layer, device):
        tdct = torch.load(pretrained_model_path, map_location=device)
         
        for key in list(tdct.keys()):
            if drop_layer in key:
                tdct.pop(key) 
        return tdct
    
    def forward(self, x):
        out = self.features(x)
        return out

    
    
    
#loss function
class Focal_loss(nn.Module):
    def __init__(self, num_class=2, alpha=0.6, gamma=2, balance_index=0, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class 
        self.alpha = alpha 
        self.gamma = gamma 
        self.smooth = smooth 
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
    
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss 
