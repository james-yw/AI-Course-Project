import copy
import torch
import numpy as np
import torch.nn as nn

class GVS(object):
    def __init__(self):
        # changing parameters
        
        
        self.cpu = False #self.cpu=True, default training on CPU, else training on GPU
        self.gpu = "0"  # the number represent the GPU device
        
        self.batch_size = 150
        self.max_max_epoch = 100
        self.exp_target = "Malignant"
        
        # fixing parameters
        
        #Network Model, we provide AlexNet, MobileNetv1, MobileNetv1, MobileNetv2, MobileNetv3_small, MobileNetv3_large, squeezeNet1_0, squeezeNet1_1
        
#         self.net = "AlexNet"
#         self.net = "squeezenet"
        self.net = "mobilenet_v1"
#         self.net = "mobilenet_v2"
#         self.net = "mobilenet_v3_small"
#         self.net = "mobilenet_v3_large"
        self.net = "convnet"
        
        #input_shape & output_shape
        self.width = 256
        self.height = 256
        self.channel = 3
        self.num_classes = 10
        
        self.input_dim = (self.batch_size, self.channel, self.height, self.width)
        self.output_dim = (self.batch_size, self.num_classes)
        
        
        #Optimizer, we provide adadelta, SGD, Adam, AdaGrad, RMSProp
        self.optimizer = "adadelta"
#         self.optimizer = "SGD"
#         self.optimizer = "Adam"
#         self.optimizer = "AdaGrad"
#         self.optimizer = "RMSProp"
        
        
        #Loss Function, Support CrossEntropyLoss, Focal_loss
        self.loss = 'CrossEntropyLoss'
#         self.loss = 'Focal_loss'
        
        
        #Checkpoint_path
        self.checkpoint_path = self.net + '_epoch' + str(self.max_max_epoch) + '_' + self.optimizer
        
        
        #Learning rate & Learning Decay
        
        self.learning_rate = 0.01 #learning rate
        self.decay_rate = 1       #0-1, if decay_rate = 1, no decay
        self.decay_step = 10      #when epoch % decay_step ==0, learning_rate = learning_rate*decay_rate
        
        #load pretrained_model
        self.pretrain = False     #self.pretrain = False, Don't use pre-trained model
#         self.pretrained_model_path = './pretrain/alexnet-pytorch-1.pth'  #pre-trained model path, should correlate to network model
        
        self.pretrained_model_path = '../pretrain/squeezenet1_0.pth'
        
        
        self.drop_layer = 'classifier.1'   #When loading pre-trained model parameters, the drop layer represent which layer do not load pre-trained model parameters
        
        
        
        ########
        #  How to train the model
        #  python train.py
        ########
        
        
        