import torch
from torch import nn
import torchsummary
import torch.nn.functional as F
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.dropout=nn.Dropout(0.5)
        self.block1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.block2=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.block3=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.block4=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.block5=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.block6=nn.Sequential(nn.Flatten(),
                                  nn.Linear(7*7*512,4096),
                                  nn.ReLU())
        self.block7=nn.Sequential(nn.Linear(4096,4096),
                                  nn.ReLU())
        self.block8=nn.Sequential(nn.Linear(4096,10))
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        x=self.dropout(x)
        x=self.block7(x)
        x=self.dropout(x)
        x=self.block8(x)
        return x
if __name__=='__main__':
    device=torch.device('cuda')
    model=VGG()
    model=model.to(device)
    print(torchsummary.summary(model,(1,224,224)))
