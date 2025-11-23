from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import VGG
import torch
from torch import nn
import time
import copy
import pandas as pd
def train_data_process():
    train_data=FashionMNIST(root='D:/Code/pytorch/LeNet/data',train=True,transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),download=True)
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader=Data.DataLoader(dataset=train_data,batch_size=128,shuffle=True,num_workers=0)
    val_dataloader=Data.DataLoader(dataset=val_data,batch_size=128,shuffle=True,num_workers=0)
    return train_dataloader,val_dataloader
def train_model_process(model,train_data_loader,val_data_loader,epochs):
    device=torch.device('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()
    model=model.to(device)
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]
    since=time.time()
    for epoch in range(epochs):
        print(f'epoch{epoch}:')
        print('-'*15)
        train_loss=0.0
        val_loss=0.0
        train_corrects=0
        val_corrects=0
        train_num=0
        val_num=0
        model.train()  
        for d_x,d_y in train_data_loader:
            d_x=d_x.to(device,non_blocking=True)
            d_y=d_y.to(device,non_blocking=True)
            output=model(d_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,d_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*d_x.size(0)
            train_corrects+=torch.sum(pre_lab==d_y.data)
            train_num+=d_x.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append((train_corrects / train_num).item()) 
        model.eval()   
        with torch.no_grad(): 
            for d_x,d_y in val_data_loader:
                d_x=d_x.to(device,non_blocking=True)
                d_y=d_y.to(device,non_blocking=True)
                output=model(d_x)
                loss=criterion(output,d_y)
                pre_lab=torch.argmax(output,dim=1)
                val_loss+=loss.item()*d_x.size(0)
                val_corrects+=torch.sum(pre_lab==d_y.data)
                val_num+=d_x.size(0)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append((val_corrects / val_num).item()) 
        
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        end=time.time()
        print(f'train loss:{train_loss_all[-1]} val loss:{val_loss_all[-1]}')
        print(f'train acc:{train_acc_all[-1]} val acc:{val_acc_all[-1]}')
        print(f'consuming time:{end-since}')
    
    model.load_state_dict(best_model_wts)
    torch.save(model,'best_model.pth')
    train_process=pd.DataFrame(data={'epoch':range(epochs),
                                     'train_loss_all':train_loss_all,
                                     'val_loss_all':val_loss_all,
                                     'train_acc_all':train_acc_all,
                                     'val_acc_all':val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_process.epoch,train_process.train_loss_all,'ro-',label='train loss')
    plt.plot(train_process.epoch,train_process.val_loss_all,'bo-',label='val loss')  
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(train_process.epoch,train_process.train_acc_all,'ro-',label='train acc')
    plt.plot(train_process.epoch,train_process.val_acc_all,'bo-',label='val acc')    
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()
if __name__=='__main__':
    train_data_loader,val_data_loader=train_data_process()
    model=VGG()
    train_process=train_model_process(model,train_data_loader,val_data_loader,20)
    matplot_acc_loss(train_process)
    while True:
        user_input=input()
        if user_input.strip().lower()=='q':
            break


    


               



