import torch
import torchsummary
from torch import nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig=nn.Sigmoid()
        self.s1=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten=nn.Flatten()
        self.f1=nn.Linear(400,120)
        self.f2=nn.Linear(120,84)
        self.f3=nn.Linear(84,10)
    def forward(self,x):
        x=self.s1(self.sig(self.c1(x)))
        x=self.s2(self.sig(self.c2(x)))
        x=self.flatten(x)
        x=self.f3(self.f2(self.f1(x)))
        return x
if __name__=="__main__":
    device=torch.device('cuda')
    model=LeNet().to(device)
    print(torchsummary.summary(model,(1,28,28)))