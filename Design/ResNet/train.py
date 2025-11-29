import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
from model import ResNet18
from emo_dataset import get_emotion_dataloader
def train():
    data_dir='D:/Code/pytorch/Design/ResNet/Fer2013'
    num_epochs=60
    batch_size=128
    lr=0.01
    device=torch.device('cuda')
    dataloaders,datasize,class_names=get_emotion_dataloader(data_dir,batch_size)
    model=ResNet18(num_classes=7,in_channels=1)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    opti=optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    scheduler=optim.lr_scheduler.StepLR(opti,step_size=20,gamma=0.1)
    best_acc=0.0
    since=time.time()
    for epoch in range(num_epochs):
        print(f'epoch:{epoch}')
        print('-'*20)
        for phase in ['train','val']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            epoch_loss=0.0
            epoch_corrects=0
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                opti.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                    if phase=='train':
                        loss.backward()
                        opti.step()
                epoch_loss+=loss.item()*inputs.size(0)
                epoch_corrects+=torch.sum(preds==labels.data)
            if phase=='train':
                scheduler.step()
            epoch_loss=epoch_loss/datasize[phase]
            epoch_acc=epoch_corrects.double()/datasize[phase]
            print(f'{phase} loss:{epoch_loss:.4f} acc:{epoch_acc:.4f}')
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                torch.save(model.state_dict(),'D:/Code/pytorch/Design/resnet.pth')
            print(f'consuming time:{time.time()-since}')
if __name__=='__main__':
    train()
