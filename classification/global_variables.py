import copy
import torch
import numpy as np
import torch.nn as nn

class GVS(object):
    def __init__(self):
        # changing parameters
        
        self.cpu = False
        self.gpu = "0"
        self.batch_size = 150
        self.max_max_epoch = 100
        self.exp_target = "Malignant"
        
        # fixing parameters
        
        #Network Model, we provide AlexNet, MobileNetv1, MobileNetv1, MobileNetv2, MobileNetv3_small, MobileNetv3_large, squeezeNet1_0, squeezeNet1_1
        
        self.net = "AlexNet"
        
        #input_shape & output_shape
        self.width = 256
        self.height = 256
        self.channel = 3
        self.num_classes = 10
        
        self.input_dim = (self.batch_size, self.channel, self.height, self.width)
        self.output_dim = (self.batch_size, self.num_classes)
        
        
        #Optimizer, we provide adadelta, SGD, Adam, AdaGrad, RMSProp
        self.optimizer = "adadelta"
        
        #Loss Function, Support CrossEntropyLoss, Focal_loss
        #self.loss = 'CrossEntropyLoss'
        
        
        #Checkpoint_path
        self.checkpoint_path = self.net + '_epoch' + str(self.max_max_epoch) + '_' + self.optimizer
        
        
        #Learning rate & Learning Decay
        self.learning_rate = 0.01
        self.decay_rate = 1
        self.decay_step = 10
        
        #load pretrained_model
        self.pretrain = True
        self.pretrained_model_path = './pretrain/alexnet-pytorch-1.pth'
        self.drop_layer = 'classifier.6'
        
        
        