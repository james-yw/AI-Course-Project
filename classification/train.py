import torch
import torch.nn as nn
import datetime, time, os, logging
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from global_variables import GVS
from net_model import NetModel

from torchstat import stat


tmp_gvs = GVS()
log_filename_prefix = "Train_" + tmp_gvs.exp_target + "_Test" + "_" + tmp_gvs.net.replace(" ", "_") + "_"
os.environ['TZ'] = 'UTC-8'
time.tzset()
date_and_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log_filepath = os.path.join("Logs", tmp_gvs.checkpoint_path)
if not os.path.exists(log_filepath):
    os.mkdir(log_filepath)
log_filename = log_filepath +'/'+ log_filename_prefix + date_and_time + ".log"
if not os.path.exists("Logs"): os.makedirs("Logs")
if os.path.exists(log_filename): os.remove(log_filename)

logger = logging.getLogger()
fh = logging.FileHandler(log_filename)
ch = logging.StreamHandler()

fh.setLevel(logging.INFO)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %A %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
    
def main():
    
    train_gvs, test_gvs, train_loader, test_loader = load_data_cifar10()
    print("Model:",train_gvs.net)
    if train_gvs.pretrain:
        print("Using pretrain model, model path:",train_gvs.pretrained_model_path)
    model = NetModel(train_gvs)
    
    ############################
    #Type the parameter and mult-ADD on Screen
    ############################ 
    data = next(iter(train_loader))[0]
    model.model_param(data)



    max_valid_acc = -1.0
    max_valid_epoch = -1
    min_valid_fnr = 2.0
    min_valid_fnr_epoch = -1
    print("Begin Training")

    logger.info("Begin Training")
    for epoch in range(train_gvs.max_max_epoch):
        
        ############################
        #learning rate decay
        ############################
        if epoch%train_gvs.decay_step==0:
            train_gvs.learning_rate = train_gvs.learning_rate* train_gvs.decay_rate
            
        logger.info("Epoch:%4d, learning rate = %.4f" % (epoch + 1, model.optimizer.defaults["lr"]))
        print("Begin Training")
        logger.info("Begin Training")
        
        train_acc, train_loss = train(train_gvs, model, train_loader, epoch + 1)
        valid_acc, valid_loss = valid(test_gvs, model, test_loader, epoch + 1)

        logger.info("Epoch:%4d, Train Accuracy: %.4f" % (epoch + 1, train_acc))
        logger.info("Epoch:%4d, Valid Accuracy: %.4f" % (epoch + 1, valid_acc))
        model.save_params(epoch + 1)


def train(gvs, model, data_loader, epoch):
    batch_size = gvs.batch_size
    all_cost = []
    count = 0
    for sidx, [data, label] in enumerate(data_loader, 0):
        error, model_output = model.train(data, label)

        all_cost.append(error)
        count += model.is_correct(label, model_output)

        if (sidx + 1) % 10 == 0:
            logger.info(
                "[Train] Epoch:%4d, BatchIdx:%4d, Cost:%.6f, Acc:%.4f" % (
                epoch, sidx + 1, np.mean(all_cost), count / ((sidx + 1) * batch_size)))
    return count / ((sidx + 1) * batch_size), np.mean(all_cost)


def valid(gvs, model, data_loader, epoch):
    batch_size = gvs.batch_size
    all_cost = []
    count = 0
    for sidx, [data, label] in enumerate(data_loader, 0):
        error, model_output = model.predict(data, label)

        all_cost.append(error)
        count += model.is_correct(label, model_output)

        if (sidx + 1) % 10 == 0:
            logger.info(
                "[Valid] Epoch:%4d, BatchIdx:%4d, Cost:%.6f, Acc:%.4f" % (
                epoch, sidx + 1, np.mean(all_cost), count / ((sidx + 1) * batch_size)))
    return count / ((sidx + 1) * batch_size), np.mean(all_cost)


def load_data_cifar10():
    train_gvs = GVS()
    test_gvs = GVS()

    data_transforms = transforms.Compose(
        [transforms.Resize([train_gvs.width, train_gvs.height]), transforms.ToTensor()])

    train_dataset = datasets.CIFAR10('../Datasets', train=True, download=True, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=train_gvs.batch_size, num_workers=0, drop_last=False,
                              shuffle=True)

    test_dataset = datasets.CIFAR10(root='../Datasets', train=False, download=True, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_gvs.batch_size, shuffle=False,
                                              num_workers=0)
    return train_gvs, test_gvs, train_loader, test_loader

if __name__ =="__main__":
    main()