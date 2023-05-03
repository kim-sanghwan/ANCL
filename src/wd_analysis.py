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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Average(lst):
    return sum(lst) / len(lst)

def Get_WD(model1, model2):
    model1.eval()
    model2.eval()

    diff = 0
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            # exclude parameters from batch normalization layer
            continue
        diff += torch.sum((model2.state_dict()[name].data - model1.state_dict()[name].data)**2).item()
    return math.sqrt(diff)





if __name__ == "__main__":
    """
    In order to run this code, you need to train and save model (.pth) in the corresponding directory in advance.
        Auxiliary Network: "./models/normal/auxiliary_network_{task+1}.pth",
        Old Network: "./models/normal/old_network_{task}.pth",
        Joint Network: "./models/normal/joint_network_{task+1}.pth".
        ANCL Networks: "./models/ANCL/{keyword_ANCL}/ANCL_network{task+1}_lamb{lamba}.pth"
    """
    tstart = time.time()

    parser = argparse.ArgumentParser(description='Generating Weight Difference Figure')

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

    for task in range(1): # WD is calculated between task 0 and 1. You can increase this value to get WD on later task.
        trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets, num_tasks, nc_first_task,
                                                                      batch_size, num_workers=num_workers,
                                                                      pin_memory=pin_memory)

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

        total_results = {"old_net": [], "aux_net": [], "joint": []}
        print("ANCL and old_net")
        for ANCL_model in ANCL_model_list:
            Weight_diff = Get_WD(ANCL_model, old_net)
            Weight_diff = round(Weight_diff, 4)
            total_results["old_net"].append(Weight_diff)

        print("ANCL and aux_net")
        for ANCL_model in ANCL_model_list:
            Weight_diff = Get_WD(ANCL_model, aux_net)
            Weight_diff = round(Weight_diff, 4)
            total_results["aux_net"].append(Weight_diff)

        print("ANCL and joint")
        for ANCL_model in ANCL_model_list:
            Weight_diff = Get_WD(ANCL_model, joint_model)
            Weight_diff = round(Weight_diff, 4)
            total_results["joint"].append(Weight_diff)


        new_lamba_list = [float(val) for val in lamba_list] 

        x_len = len(lamba_list)

        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        
        ax1.plot(new_lamba_list, total_results["old_net"], label = "dist(ANCL, old)", marker='o',linestyle='dashed')
        ax1.plot(new_lamba_list, total_results["aux_net"], label = "dist(ANCL, aux)", marker='o', linestyle='dashed')

        ax1.set_xscale('log')
        plt.grid()
        plt.savefig(f'./figures/wd/{keyword_ANCL}/wd_plot_{task}{task+1}.jpg')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')