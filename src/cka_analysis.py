from datasets.data_loader import get_loaders
import torch
import torch.nn as nn

from networks.network import LLL_Net
from networks.resnet32 import resnet32

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from torch.utils.data import DataLoader, Dataset
import time
import pickle

from matplotlib import pyplot as plt


activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def Average(lst):
    return sum(lst) / len(lst)

def get_features(model, images, layer_name):
    outputs = model(images.to(device))
    feature_dict = {}
    for name in layer_name:
        feature_dict[name] = activation[name].cpu()
    return feature_dict

def register_hooks(model, layer_name):
    """
        CKA is calculated on the last resnet block to represent more reliable output similarity.
        If you want to calculate CKA using whole block output, uncomment below codes (It does not change the general trend from our experiment).
    """

    """
    #First convolutional layer
    model.model.conv1.register_forward_hook(get_activation('conv1.weight'))
    layer_name.append('conv1.weight')

    #convolutional layers for layer1
    for i in range(5):
        model.model.layer1[i].conv1.register_forward_hook(get_activation('layer1.{}.conv1.weight'.format(i)))
        model.model.layer1[i].conv2.register_forward_hook(get_activation('layer1.{}.conv2.weight'.format(i)))
        layer_name.append('layer1.{}.conv1.weight'.format(i))
        layer_name.append('layer1.{}.conv2.weight'.format(i))

    #convolutional layers for layer2
    for i in range(5):
        model.model.layer2[i].conv1.register_forward_hook(get_activation('layer2.{}.conv1.weight'.format(i)))
        model.model.layer2[i].conv2.register_forward_hook(get_activation('layer2.{}.conv2.weight'.format(i))) 
        layer_name.append('layer2.{}.conv1.weight'.format(i))
        layer_name.append('layer2.{}.conv2.weight'.format(i)) 
    """

    #convolutional layers for layer3
    for i in range(5):
        model.model.layer3[i].conv1.register_forward_hook(get_activation('layer3.{}.conv1.weight'.format(i)))
        model.model.layer3[i].conv2.register_forward_hook(get_activation('layer3.{}.conv2.weight'.format(i)))          
        layer_name.append('layer3.{}.conv1.weight'.format(i))
        layer_name.append('layer3.{}.conv2.weight'.format(i)) 
    return layer_name

# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


def Get_CKA(model1, model2, trn_loader):
    model1.eval()
    model2.eval()

    layer_name = register_hooks(model1, [])
    _ = register_hooks(model2, []) # output is same as layer_name

    for i, (images, targets) in enumerate(trn_loader):
        if i ==0:
            feature_dict1 = get_features(model1, images, layer_name)
            feature_dict2 = get_features(model2, images, layer_name)
        else:
            temp_dict1 = get_features(model1, images, layer_name)
            temp_dict2 = get_features(model2, images, layer_name)
            for name in layer_name:
                feature_dict1[name] = torch.cat((feature_dict1[name], temp_dict1[name]), 0)
                feature_dict2[name] = torch.cat((feature_dict2[name], temp_dict2[name]), 0)

    data_num = feature_dict1[name].shape[0]
    for name in layer_name:
        feature_dict1[name] = torch.reshape(feature_dict1[name], (data_num, -1))
        feature_dict2[name] = torch.reshape(feature_dict2[name], (data_num, -1))

    CKA = CudaCKA(device)

    cka_list = []

    for name in layer_name:
        cka_value = CKA.linear_CKA(feature_dict1[name].to(device), feature_dict2[name].to(device))
        cka_list.append(round(cka_value.item(),4))

    return cka_list


if __name__ == "__main__":
    """
    In order to run this code, you need to train and save model (.pth) in the corresponding directory in advance.
        Auxiliary Network: "./models/normal/auxiliary_network_{task+1}.pth",
        Old Network: "./models/normal/old_network_{task}.pth",
        Joint Network: "./models/normal/joint_network_{task+1}.pth".
        ANCL Networks: "./models/ANCL/{keyword_ANCL}/ANCL_network{task+1}_lamb{lamba}.pth"
    """
    tstart = time.time()

    parser = argparse.ArgumentParser(description='Generating CKA Figure')

    parser.add_argument('--task', type=int, default=0,
                            help='Which task to evaluate on.')

    args = parser.parse_args() 

    task = args.task

    datasets = ['cifar100_icarl']
    num_tasks = 10
    nc_first_task = None
    batch_size = 128
    num_workers = 2
    pin_memory = False
    multi_softmax = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyword_CL="ewc"
    keyword_ANCL= "A-" + keyword_CL

    #EWC
    lamb = "10000.0"
    lamba_list = ["10.0", "100.0", "1000.0", "10000.0", "20000.0", "40000.0"]

    """
    #MAS
    lamb = "50.0"
    lamba_list = ["1.0", "5.0", "10.0", "50.0",  "100.0", "200.0"]

    #LwF
    lamb = "10.0"
    lamba_list = ["0.05", "0.1", "0.5", "1.0", "5.0", "10.0"]

    #LFL
    lamb = "400.0"
    lamba_list = ["10.0", "50.0", "100.0", "200.0", "400.0", "800.0"]
    """

    for task in range(1): # CKA is calculated between task 0 and 1. You can increase this value to get CKA on later task.
        trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets, num_tasks, nc_first_task,
                                                                      batch_size, num_workers=num_workers,
                                                                      pin_memory=pin_memory)
        print("-"*108)
        print(f"Current task: {task}")

        """Load Auxiliary Network"""
        init_model = resnet32()
        aux_net = LLL_Net(init_model, remove_existing_head= True)
        for _ in range(task+2):
            aux_net.add_head(10)
        aux_net.load_state_dict(torch.load(f"./models/normal/auxiliary_network_{task+1}.pth"\
                                , map_location=device))
        aux_net.to(device)
        aux_net.eval()

        """Load Old Network"""
        init_model = resnet32()
        old_net = LLL_Net(init_model, remove_existing_head= True)
        for _ in range(task+1):
            old_net.add_head(10)
        old_net.load_state_dict(torch.load(f"./models/normal/old_network_{task}.pth"\
                                , map_location=device))
        old_net.add_head(10)
        # Use same head as aux_net for task t+1
        with torch.no_grad():
            for name, param in old_net.state_dict().items(): 
                if ('heads.{}.'.format(task+1) in name):
                    param.copy_(aux_net.state_dict()[name].data)  
        old_net.to(device)
        old_net.eval()

        """Load Joint(multitask) Network"""
        init_model = resnet32()
        joint_model = LLL_Net(init_model, remove_existing_head= True)
        for _ in range(task+2):
            joint_model.add_head(10)
        joint_model.load_state_dict(torch.load(f"./models/normal/joint_network_{task+1}.pth"\
                                , map_location=device))
        joint_model.to(device)
        joint_model.eval()

        """Load ANCL Models with different Lambda_a"""
        ANCL_model_list = []
        for lamba in lamba_list:   
            init_model = resnet32()
            ANCL_model = LLL_Net(init_model, remove_existing_head= True)
            for _ in range(task+2):
                ANCL_model.add_head(10)
            ANCL_model.load_state_dict(torch.load(f"./models/ANCL/{keyword_ANCL}/ANCL_network{task+1}_lamb{lamba}.pth"\
                                    , map_location=device))
            ANCL_model.to(device)
            ANCL_model.eval()
            ANCL_model_list.append(ANCL_model)

        total_results = {"old_net": [], "aux_net": [], "joint": [], "old_net_mean": [], "aux_net_mean": [], "joint_mean": []}
        print("ANCL and old_net")
        for ANCL_model in ANCL_model_list:
            cka_list = Get_CKA(ANCL_model, old_net, trn_loader[task])
            mean = round(Average(cka_list),4)
            print(cka_list)
            print(f"mean : {mean}")
            total_results["old_net"].append(cka_list)
            total_results["old_net_mean"].append(mean)

        print("ANCL and aux_net")
        for ANCL_model in ANCL_model_list:
            cka_list = Get_CKA(ANCL_model, aux_net, trn_loader[task])
            mean = round(Average(cka_list),4)
            print(cka_list)
            print(f"mean : {mean}")
            total_results["aux_net"].append(cka_list)
            total_results["aux_net_mean"].append(mean)

        print("ANCL and joint")
        for ANCL_model in ANCL_model_list:
            cka_list = Get_CKA(ANCL_model, joint_model, trn_loader[task])
            mean = round(Average(cka_list),4)
            print(cka_list)
            print(f"mean : {mean}")
            total_results["joint"].append(cka_list)
            total_results["joint_mean"].append(mean)
        print("-"*108)

        #Save Figure
        new_lamba_list = [float(val) for val in lamba_list]    
        ANCL_old_net = total_results["old_net_mean"]
        ANCL_aux_net = total_results["aux_net_mean"]
        ANCL_joint = total_results["joint_mean"]
        print(ANCL_old_net)
        print(ANCL_aux_net)
        print(ANCL_joint)
        
        x_len = len(ANCL_old_net)

        plt.figure(figsize=(5, 3.5))

        plt.plot(new_lamba_list, ANCL_old_net, label = 'ANCL_old_net', marker="o", linestyle='dashed')
        plt.plot(new_lamba_list, ANCL_aux_net, label = 'ANCL_aux_net' , marker="o", linestyle='dashed')
        plt.plot(new_lamba_list, DCL_joint, label = 'ANCL_joint' , marker="o", linestyle='dashed')

        plt.grid()
        plt.xscale("log")
        #plt.legend()
        plt.savefig(f'./figures/cka/{keyword_ANCL}/cka_plot_{task}{task+1}.jpg')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')