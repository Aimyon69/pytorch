from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet
import torch
from torch import nn
import time
import copy
import pandas
def train_val_data_process():
    train_data=FashionMNIST(root='./data',train=True,transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]),download=True)
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader=Data.DataLoader(dataset=train_data,batch_size=128,shuffle=True,num_workers=8)
    val_dataloader=Data.DataLoader(dataset=val_data,batch_size=128,shuffle=True,num_workers=8)
    return train_dataloader,val_dataloader
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
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
    for epoch in range(num_epochs):
        print(f'Epoch{epoch}')
        print('-'*10)
        train_loss=0.0
        train_corrects=0
        val_loss=0.0
        val_corrects=0
        train_num=0
        val_num=0
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.train()
            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*b_x.size(0)
            train_corrects+=torch.sum(pre_lab==b_y.data)
            train_num+=b_x.size(0)
        for step,(b_x,b_y) in enumerate(val_dataloader):
             b_x=b_x.to(device)
             b_y=b_y.to(device)
             model.eval()
             output=model(b_x)
             pre_lab=torch.argmax(output,dim=1)
             loss=criterion(output,b_y)
             val_loss+=loss.item()*b_x.size(0)
             val_corrects+=torch.sum(pre_lab==b_y.data)
             val_num+=b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print('{} train loss:{:.4f} train acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} val loss:{:.4f} val acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        time_use=time.time()-since
        print(f'consuming time:{time_use}')
        torch.save(model.load_state_dict(best_model_wts),'best_model.pth')
    train_process=pandas.DataFrame(data={'epoch':range(num_epochs),
                                             'train_loss_all':train_loss_all,
                                             'val_loss_all':val_loss_all,
                                             'train_acc_all':train_acc_all,
                                             'val_acc_all':val_acc_all})
    return train_process
if __name__=='__main__':
    LeNet=LeNet()
    train_dataloader,val_dataloader=train_val_data_process()
    train_process=train_model_process(LeNet,train_dataloader,val_dataloader,20)
    


