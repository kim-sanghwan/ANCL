from datasets.data_loader import get_loaders
import torch
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
from scipy.ndimage.filters import gaussian_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y

#model1----multitask_model----model2
def loss_contour_map(model1, model2, multitask_model, CL_model, ANCL_model_list, t, total_trn_loader, lamba_list, keyword_CL, keyword_ANCL):
    #calculate weight vector model2-model1 and set it as axis x direction
    x_diff = 0
    x_param_list = {}
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            # exclude parameters from batch normalization layer
            continue
        x_param_list[name] = model2.state_dict()[name].data - model1.state_dict()[name].data
        x_diff += torch.sum(x_param_list[name]**2).item()
    
    #calculate weight vector multitask_model-model1 and set is as temp direction
    temp_diff = 0
    temp_param_list = {}
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue
        temp_param_list[name] = multitask_model.state_dict()[name].data - model1.state_dict()[name].data
        temp_diff += torch.sum(temp_param_list[name]**2).item()
    
    #calculate y axis given x and temp vector.  
    dot_product = 0
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue
        dot_product += torch.sum(temp_param_list[name] * x_param_list[name]).item()
    
    y_diff = 0
    x_pos = 0
    y_param_list = {}
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue
        y_param_list[name] = temp_param_list[name] - (dot_product/x_diff)* x_param_list[name]
        y_diff += torch.sum(y_param_list[name]**2).item()
        x_pos += torch.sum(((dot_product/x_diff)* x_param_list[name])**2).item()


    #Sanity check to see x and y axis is valid
    should_zero = 0
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue 
        should_zero += torch.sum(x_param_list[name] * y_param_list[name]).item() 
    print(f"should_zero {should_zero}", file = save_stdout)
    assert x_pos <= x_diff


    # Get projection coordinate
    CL_xdot_product = 0
    CL_ydot_product = 0
    CL_param_list = {}
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue
        CL_param_list[name] = CL_model.state_dict()[name].data - model1.state_dict()[name].data
        CL_xdot_product += torch.sum(CL_param_list[name] * x_param_list[name]).item()
        CL_ydot_product += torch.sum(CL_param_list[name] * y_param_list[name]).item()     

    CL_x_pos = 0
    CL_y_pos = 0
    CL_left_param_diff = 0
    CL_left_param_list = {}
    for name in model1.state_dict().keys(): 
        if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
            continue
        CL_x_pos += torch.sum(((CL_xdot_product/x_diff)* x_param_list[name])**2).item()
        CL_y_pos += torch.sum(((CL_ydot_product/y_diff)* y_param_list[name])**2).item()

        CL_left_param_list[name] = CL_param_list[name] - (CL_xdot_product/x_diff)* x_param_list[name] \
                                        - (CL_ydot_product/y_diff)* y_param_list[name]
        CL_left_param_diff += torch.sum(CL_left_param_list[name]**2).item()

    CL_x_pos = math.sqrt(CL_x_pos)
    CL_y_pos = math.sqrt(CL_y_pos)
    CL_left_param_diff = math.sqrt(CL_left_param_diff)


    ANCL_info_list = []   
    for ANCL_model in ANCL_model_list:
        ANCL_xdot_product = 0
        ANCL_ydot_product = 0
        ANCL_param_list = {}
        for name in model1.state_dict().keys(): 
            if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
                continue
            ANCL_param_list[name] = ANCL_model.state_dict()[name].data - model1.state_dict()[name].data
            ANCL_xdot_product += torch.sum(ANCL_param_list[name] * x_param_list[name]).item()
            ANCL_ydot_product += torch.sum(ANCL_param_list[name] * y_param_list[name]).item()     

        ANCL_x_pos = 0
        ANCL_y_pos = 0
        ANCL_left_param_diff = 0
        ANCL_left_param_list = {}
        for name in model1.state_dict().keys(): 
            if ('num_batches_tracked' in name) or ('running_var' in name) or ('running_mean' in name):
                continue
            ANCL_x_pos += torch.sum(((ANCL_xdot_product/x_diff)* x_param_list[name])**2).item()
            ANCL_y_pos += torch.sum(((ANCL_ydot_product/y_diff)* y_param_list[name])**2).item()

            ANCL_left_param_list[name] = ANCL_param_list[name] - (ANCL_xdot_product/x_diff)* x_param_list[name] \
                                            - (ANCL_ydot_product/y_diff)* y_param_list[name]
            ANCL_left_param_diff += torch.sum(ANCL_left_param_list[name]**2).item()
        ANCL_x_pos = math.sqrt(ANCL_x_pos)
        ANCL_y_pos = math.sqrt(ANCL_y_pos)
        ANCL_left_param_diff = math.sqrt(ANCL_left_param_diff)
        ANCL_info_list.append((ANCL_x_pos, ANCL_y_pos, ANCL_left_param_diff))

    #Calculation results
    x_diff = math.sqrt(x_diff)
    temp_diff = math.sqrt(temp_diff)
    x_pos = math.sqrt(x_pos)
    y_diff = math.sqrt(y_diff)
    print(f"x_diff {x_diff}", file = save_stdout) 
    print(f"temp_diff {temp_diff}", file = save_stdout)
    print(f"dot_product {dot_product}", file = save_stdout)
    print(f"y_diff {y_diff}", file = save_stdout)
    print(f"x_pos {x_pos}", file = save_stdout)
    print(f"model {t} : (0, 0)", file = save_stdout)
    print(f"model {t+1} : ({round(x_diff,2)}, 0)", file = save_stdout)
    print(f"Joint model {t+1} : ({round(x_pos,2)}, {round(y_diff,2)})", file = save_stdout)
    print(f"{keyword_CL} : ({round(CL_x_pos,2)}, {round(CL_y_pos,2)}), left vector length: {CL_left_param_diff}", file = save_stdout)
    for lamba, ANCL_info in zip(lamba_list, ANCL_info_list):
        print(f"lamba {lamba} {keyword_ANCL} : ({round(ANCL_info[0],2)}, {round(ANCL_info[1],2)}), left vector length: {round(ANCL_info[2],2)}", file = save_stdout)


    #Divide subspace with 100*100 points
    xlist = np.linspace(-3/10*x_diff, 13/10*x_diff, 100)
    ylist = np.linspace(-3/10*y_diff, 13/10*y_diff, 100)
    X, Y = np.meshgrid(xlist, ylist)

    Z1 = np.random.randn(100,100) #Task t loss landscape
    Z1_2 = np.random.randn(100,100) #Task t acc landscape

    Z2 = np.random.randn(100,100) #Task t+1 loss landscape
    Z2_2 = np.random.randn(100,100) #Task t+1 acc landscape

    Z3 = np.random.randn(100,100) #Task 1~t+1 mean loss landscape
    Z3_2 = np.random.randn(100,100)  #Task 1~t+1 mean acc landscape   

    init_model = resnet32()
    model_temp = LLL_Net(init_model, remove_existing_head= True)
    for _ in range(t+2):
        model_temp.add_head(10) 
    model_temp.to(device)   
    model_temp.eval()


    #calculate loss and accuracy at 100*100 points and save it for later. 
    total_results = []
    for y_tick in tqdm(range(100)):
        x_results = []
        for x_tick in range(100):
            with torch.no_grad():
               for name, param in model_temp.state_dict().items(): 
                    if ('num_batches_tracked' in name) or ('running_mean' in name):
                        param.zero_()
                    elif 'running_var' in name:
                        param.fill_(1)
                    else:
                        param.copy_(model1.state_dict()[name].data + xlist[x_tick]/x_diff* x_param_list[name] \
                        + ylist[y_tick]/y_diff* y_param_list[name])   
        
            model_temp.train()
            for images, targets in total_trn_loader:
                # Forward current model to update running_mean and running_var of batchnorm
                outputs = model_temp(images.to(device))
            model_temp.eval()

            point_results = []
            for task in range(t+2):
                result_temp = eval(model_temp, task, tst_loader[task])  
                point_results.append(result_temp)
            x_results.append(point_results)          
            #print(point_results)
            #Caculate mean acc and loss so far
            tot_loss = 0
            tot_acc = 0 
            for res in point_results[:t+1]:
                tot_loss += res[0]
                tot_acc += res[1]


            Z1[y_tick, x_tick] = tot_loss/(t+1)
            Z1_2[y_tick, x_tick] = tot_acc/(t+1)

            Z2[y_tick, x_tick] = point_results[t+1][0]
            Z2_2[y_tick, x_tick] = point_results[t+1][1]

            Z3[y_tick, x_tick] = (tot_loss + point_results[t+1][0])/(t+2)
            Z3_2[y_tick, x_tick] = (tot_acc + point_results[t+1][1])/(t+2)

        total_results.append(x_results)

    with open(f"./figures/mean_acc/subspace_loss_acc{t}{t+1}.txt", "wb") as fp:   #Pickling
        pickle.dump(total_results, fp)   

    """
    #Load previously saved file. Comment above for-loop if you load file.
    with open(f"./figures/mean_acc/subspace_loss_acc{t}{t+1}.txt", 'rb') as fp:
        results = pickle.load(fp)   

    for y, valy in enumerate(results):
        for x, valx in enumerate(valy):
            tot_loss = 0
            tot_acc = 0 
            for res in valx[:t+1]:
                tot_loss += res[0]
                tot_acc += res[1]

            Z1[y, x] = tot_loss/(t+1)
            Z1_2[y, x] = tot_acc/(t+1)         

            Z2[y, x] = valx[t+1][0]
            Z2_2[y, x] = valx[t+1][1]

            Z3[y, x] = (tot_loss + valx[t+1][0])/(t+2)
            Z3_2[y, x] = (tot_acc + valx[t+1][1])/(t+2)
    """

    # denoise values to make contour smooth
    Z1 = gaussian_filter(Z1, 2)
    Z1_2 = gaussian_filter(Z1_2, 2)

    Z2 = gaussian_filter(Z2, 2)
    Z2_2 = gaussian_filter(Z2_2, 2)

    Z3 = gaussian_filter(Z3, 2)
    Z3_2 = gaussian_filter(Z3_2, 2)

    # Task index starts from 1 while t starts from 0
    #Loss landscape of task t+1
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z1, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task 0~{} Mean Loss Landscape'.format(t))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}loss_landscape_task{t+1}.jpg')


    #Acc landscape of task t+1
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z1_2, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task 0~{} Mean Accuracy Landscape'.format(t))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}acc_landscape_task{t+1}.jpg')
   
    
    #Loss landscape of task t+2   
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z2, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task {} Loss Landscape'.format(t+1))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}loss_landscape_task{t+2}.jpg')

    #Acc landscape of task t+2 
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z2_2, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task {} Accuracy Landscape'.format(t+1))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}acc_landscape_task{t+2}.jpg')


    #Mean loss landscape of task 1~t+2  
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z3, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task 0~{} Mean Loss Landscape'.format(t+1))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}mean_loss_landscape.jpg')

    #Mean Acc landscape of task 1~t+2
    fig,ax=plt.subplots(figsize=(5, 3.5))
    cp = ax.contourf(X, Y, Z3_2, cmap = 'gist_rainbow', alpha=0.6)
    fig.colorbar(cp) # Add a colorbar to a plot
    #ax.set_title('Task 0~{} Mean Accuracy Landscape'.format(t+1))
    plt.plot(0, 0, 'o', c='black') 
    plt.plot(x_diff, 0, 'o', c='black') 
    plt.plot(x_pos, y_diff, 'o', c='black') 
    plt.text(0.1, 0.1,'$\u03F4_{}^{{old}}$'.format(t+1), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_diff+0.1, 0.1,'$\u03F4_{}^{{aux}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    plt.text(x_pos+0.1, y_diff+0.1,'$\u03F4_{}^{{multi}}$'.format(t+2), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal') 

    #projections
    plt.plot(CL_x_pos, CL_y_pos, 'x', c='b') 
    plt.text(CL_x_pos+0.1, CL_y_pos+0.1,f'$\u03F4_{{{keyword_CL}}}$', c='b', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
    ANCL_x_list = []
    ANCL_y_list = []
    for i, ANCL_info in enumerate(ANCL_info_list):
        ANCL_x_list.append(ANCL_info[0])
        ANCL_y_list.append(ANCL_info[1])
    plt.plot(ANCL_x_list[1:-1], ANCL_y_list[1:-1], linestyle='--', marker='x', c='red')
    plt.plot(ANCL_x_list[:2], ANCL_y_list[:2], linestyle='--', c='red')
    plt.plot(ANCL_x_list[-2:], ANCL_y_list[-2:], linestyle='--', c='red')
    plt.plot(ANCL_x_list[0], ANCL_y_list[0], marker='o', c='saddlebrown', markersize=8)
    plt.plot(ANCL_x_list[-1], ANCL_y_list[-1], marker='*', c='saddlebrown', markersize=8)

    plt.savefig(f'./figures/mean_acc/{keyword_ANCL}/{t+1}{t+2}mean_acc_landscape.jpg')
    return


def criterion(model, t, outputs, targets):
    """Returns the loss value"""
    return torch.nn.functional.cross_entropy(outputs[t], targets - model.task_offset[t])

def calculate_metrics(model, outputs, targets):
    """Contains the main Task-Aware and Task-Agnostic metrics"""
    pred = torch.zeros_like(targets.to(device))
    # Task-Aware Multi-Head
    for m in range(len(pred)):
        this_task = (model.task_cls.cumsum(0) <= targets[m]).sum()
        pred[m] = outputs[this_task][m].argmax() + model.task_offset[this_task]
    hits_taw = (pred == targets.to(device)).float()
    # Task-Agnostic Multi-Head
    if multi_softmax:
        outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
        pred = torch.cat(outputs, dim=1).argmax(1)
    else:
        pred = torch.cat(outputs, dim=1).argmax(1)
    hits_tag = (pred == targets.to(device)).float()
    return hits_taw, hits_tag

def eval(model, t, val_loader):
    """Contains the evaluation code"""
    with torch.no_grad():
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        model.eval()
        for images, targets in val_loader:
            # Forward current model
            outputs = model(images.to(device))
            loss = criterion(model, t, outputs, targets.to(device))
            hits_taw, hits_tag = calculate_metrics(model, outputs, targets)
            # Log
            total_loss += loss.item() * len(targets)
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
    return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num


if __name__ == "__main__":
    """
    In order to run this code, you need to train and save model (.pth) in the corresponding directory in advance.
        Auxiliary Network: "./models/normal/auxiliary_network_{task+1}.pth",
        Old Network: "./models/normal/old_network_{task}.pth",
        Joint Network: "./models/normal/joint_network_{task+1}.pth".
        CL Network: "./models/CL/{keyword_CL}/CL_network{task+1}_lamb{lamb}.pth"
        ANCL Networks: "./models/ANCL/{keyword_ANCL}/ANCL_network{task+1}_lamb{lamba}.pth"
    """
    parser = argparse.ArgumentParser(description='Generating ANCL Loss Landscape Figure')

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

    for task in range(7):
        tstart = time.time()

        save_stdout = open(f"./figures/mean_acc/{keyword_ANCL}/stdout{task}{task+1}.txt", "w")

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

        """Load CL Model with optimal Lambda"""
        init_model = resnet32()
        CL_model = LLL_Net(init_model, remove_existing_head= True)
        for _ in range(task+2):
            CL_model.add_head(10)
        CL_model.load_state_dict(torch.load(f"./models/CL/{keyword_CL}/CL_network{task+1}_lamb{lamb}.pth"\
                                , map_location=device))
        CL_model.to(device)
        CL_model.eval()

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


        print("-"*108, file = save_stdout)
        print(f"Tasks of Interest : {task}, {task+1}", file = save_stdout)
        print("", file = save_stdout)
        temp = [0,0,0]
        for t in range(task+2):
            res = eval(old_net, t, tst_loader[t])
            temp[0] += res[0]
            temp[1] += res[1]     
            temp[2] += res[2]  
            print(f"old_net Eval on Task {t} : {res}", file = save_stdout)
        print(f"old_net mean on Task {t} : ({round(temp[0]/(task+2), 2)}, {round(temp[1]/(task+2), 2)}, {round(temp[2]/(task+2), 2)})" , file = save_stdout )
        
        print("", file = save_stdout)
        temp = [0,0,0]
        for t in range(task+2):
            res = eval(aux_net, t, tst_loader[t])
            temp[0] += res[0]
            temp[1] += res[1]     
            temp[2] += res[2] 
            print(f"aux_net Eval on Task {t} : {res}", file = save_stdout)
        print(f"aux_net mean on Task {t} : ({round(temp[0]/(task+2), 2)}, {round(temp[1]/(task+2), 2)}, {round(temp[2]/(task+2), 2)})" , file = save_stdout )
        
        print("", file = save_stdout)
        temp = [0,0,0]
        for t in range(task+2):
            res = eval(joint_model, t, tst_loader[t])
            temp[0] += res[0]
            temp[1] += res[1]     
            temp[2] += res[2] 
            print(f"joint_model Eval on Task {t} : {res}", file = save_stdout)
        print(f"joint_model mean on Task {t} : ({round(temp[0]/(task+2), 2)}, {round(temp[1]/(task+2), 2)}, {round(temp[2]/(task+2), 2)})" , file = save_stdout )    


        print("-"*108, file = save_stdout)
        #Projection models (CL, ANCL)
        temp = [0,0,0]
        for t in range(task+2):
            res = eval(CL_model, t, tst_loader[t])
            temp[0] += res[0]
            temp[1] += res[1]     
            temp[2] += res[2] 
            print(f"{keyword_CL} model Eval on Task {t} : {res}", file = save_stdout)
        print(f"{keyword_CL} mean on Task {t} : ({round(temp[0]/(task+2), 2)}, {round(temp[1]/(task+2), 2)}, {round(temp[2]/(task+2), 2)})" , file = save_stdout )    

        print("", file = save_stdout)
        for lamba, ANCL_model in zip(lamba_list, ANCL_model_list):
            print(f"Lamba : {lamba}", file = save_stdout)
            temp = [0,0,0]
            for t in range(task+2):
                res = eval(ANCL_model, t, tst_loader[t])
                temp[0] += res[0]
                temp[1] += res[1]     
                temp[2] += res[2] 
                print(f"{keyword_ANCL} model Eval on Task {t} : {eval(ANCL_model, t, tst_loader[t])}", file = save_stdout)
            print(f"{keyword_ANCL} mean on Task {t} : ({round(temp[0]/(task+2), 2)}, {round(temp[1]/(task+2), 2)}, {round(temp[2]/(task+2), 2)})" , file = save_stdout )    
            print("", file = save_stdout)
        print("-"*108, file = save_stdout)
        

        dataset_list = []
        for t in range(task+2):
            dataset_list.append(trn_loader[t].dataset)
        total_trn_dset = JointDataset(dataset_list)
        total_trn_loader = DataLoader(total_trn_dset,
                                      batch_size=256,
                                      shuffle=True,
                                      num_workers=2,
                                      pin_memory=False)

        loss_contour_map(old_net, aux_net, joint_model, CL_model, ANCL_model_list, task, total_trn_loader, lamba_list, keyword_CL, keyword_ANCL)

        print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)), file = save_stdout)
        print('Done!', file = save_stdout)

        save_stdout.close()
