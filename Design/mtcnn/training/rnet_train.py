import os
import sys
sys.path.append('D:/Code/pytorch/Design/mtcnn/data_process')
import time
import torch
import torch.nn as nn
from dataset import MTCNNDataset
import torch.utils.data as Data
from model_rely import RNet,get_rnet_dataloader,MTCNNLoss
import matplotlib.pyplot as plt
def train():
    device=torch.device('cuda')
    batch_size=384
    lr=0.001
    epochs=30
    train_loader=get_rnet_dataloader(batch_size)
    model=RNet()
    model=model.to(device)
    criterion=MTCNNLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        since=time.time()
        for step,(img,label,roi,pts) in enumerate(train_loader):
            img=img.to(device)
            label=label.to(device)
            roi=roi.to(device)
            pts=pts.to(device)
            cls_out,box_out,landmark_out=model(img)
            loss,l_cls,l_box,l_pts=criterion(cls_out,box_out,landmark_out,label,roi,pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f'epoch:{epoch}')
        print(f'loss:{epoch_loss/len(train_loader):.4f}')
        print(f'consuming time:{time.time()-since}')
        print('.'*20)
        scheduler.step()
    torch.save(model.state_dict(),'D:/Code/pytorch/Design/rnet.pth')
if __name__=='__main__':
    train()