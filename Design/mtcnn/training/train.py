import sys
sys.path.append(r'D:\Code\pytorch\Design\mtcnn\data_process')
import sys
import torch
import torch.nn as nn
import data_loader
class PNet(nn.Module):
    def __init__(self):
        super(PNet,self).__init__()
        self.f1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1),
                              nn.PReLU(),
                              nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
                              nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,stride=1),
                              nn.PReLU(),
                              nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),
                              nn.PReLU())
        self.classifier=nn.Conv2d(in_channels=32,out_channels=2,kernel_size=1,stride=1)
        self.bbox_reg=nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1,stride=1)
    def forward(self,x):
        x=self.f1(x)
        cls_outputs=self.classifier(x)
        bbox_outputs=self.bbox_reg(x)
        cls_outputs=nn.Flatten(cls_outputs)
        bbox_outputs=nn.Flatten(bbox_outputs)
        return cls_outputs,bbox_outputs
def train_process(model):
    device=torch.device('cuda')
    batch_size=128
    epoch=80
    lr=1e-5 
    model=PNet()
    model.to(device)
       