import torch
from torch import nn
import torch.nn.functional as F
import torchsummary
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.c1=nn.Conv2d(in_channels=1,out_channels=96,stride=4,kernel_size=11)
        self.s1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.c2=nn.Conv2d(in_channels=96,out_channels=256,padding=2,kernel_size=5)
        self.s2=nn.MaxPool2d(kernel_size=3,stride=2)
        self.c3=nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.c4=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.c5=nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.s3=nn.MaxPool2d(kernel_size=3,stride=2)
        self.flatten=nn.Flatten()
        self.f1=nn.Linear(6*6*256,4096)
        self.f2=nn.Linear(4096,4096)
        self.f3=nn.Linear(4096,10)
    def forward(self,x):
        x=self.s1(self.relu(self.c1(x)))
        x=self.s2(self.relu(self.c2(x)))
        x=self.relu(self.c3(x))
        x=self.relu(self.c4(x))
        x=self.relu(self.c5(x))
        x=self.s3(x)
        x=self.flatten(x)
        x=self.relu(self.f1(x))
        x=self.dropout(x)
        x=self.relu(self.f2(x))
        x=self.dropout(x)
        x=self.f3(x)
        return x
if __name__=='__main__':
    device=torch.device('cuda')
    model=AlexNet()
    model=model.to(device)
    print(torchsummary.summary(model,(1,227,227)))





