import sys
sys.path.append('D:/Code/pytorch/Design/mtcnn/data_process')
from dataset import MTCNNDataset
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch
def get_dataloader(batch_size=256):
    cls_path='D:/Code/pytorch/Design/mtcnn/data_process/data/cls_imdb.pkl'
    roi_path='D:/Code/pytorch/Design/mtcnn/data_process/data/roi_imdb.pkl'
    pts_path='D:/Code/pytorch/Design/mtcnn/data_process/data/pts_imdb.pkl'
    datasets=[]
    datasets.append(MTCNNDataset(cls_path))
    datasets.append(MTCNNDataset(roi_path))
    datasets.append(MTCNNDataset(pts_path))
    full_datasets=Data.ConcatDataset(datasets)
    loader=Data.DataLoader(dataset=full_datasets,batch_size=batch_size,shuffle=True,num_workers=0)
    return loader
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
        self.landmark_reg = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1)
    def forward(self,x):
        x=self.f1(x)
        cls_outputs=self.classifier(x)
        bbox_outputs=self.bbox_reg(x)
        landmark_reg_outputs=self.landmark_reg(x)
        return cls_outputs,bbox_outputs,landmark_reg_outputs
class MTCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls=nn.CrossEntropyLoss(reduction='none')
        self.loss_box=nn.MSELoss(reduction='none')
        self.loss_landmark=nn.MSELoss(reduction='none')
    def forward(self,pred_cls,pred_box,pred_landmark,gt_label,gt_box,gt_landmark):
        if pred_cls.dim() == 4:
            pred_cls=pred_cls.squeeze(3).squeeze(2)
            pred_box=pred_box.squeeze(3).squeeze(2)
            pred_landmark=pred_landmark.squeeze(3).squeeze(2)
        mask_cls=torch.ge(gt_label,0)
        valid_cls_label=gt_label[mask_cls]
        valid_cls_pred=pred_cls[mask_cls]
        '''这里我解释下上面的代码意思,因为我们训练人脸分类,只需要正样本和负样本的数据来计算loss值,
        故我们需要剔除-1对应部分样本。第一行代码就是生成掩码,筛选出部分样本。第二,三行代码就是将部分
        样本对应的模型预测值和真实标签剔除掉'''
        if valid_cls_label.shape[0]>0:
            loss_c=self.loss_cls(valid_cls_pred,valid_cls_label)
            loss_c=torch.mean(loss_c)
        else:
            loss_c=torch.tensor(0.0).to(gt_label.device)
            '''解释下,我们要凭空增加一个0张量,如果不传入原来gt_label所在的设备(GPU或者CPU)
            后面进行张量运算时，如果两个张量不在同一个设备上就会出错'''
        
        mask_box=torch.ne(gt_label,0)
        valid_box_gt=gt_box[mask_box]
        valid_box_pred=pred_box[mask_box]
        if valid_box_gt.shape[0]>0:
            loss_b=self.loss_box(valid_box_pred,valid_box_gt)
            loss_b=torch.mean(loss_b)
        else:
            loss_b=torch.tensor(0.0).to(gt_box.device)

        mask_lm=torch.sum(torch.abs(gt_landmark),dim=1)>0
        valid_lm_gt=gt_landmark[mask_lm]
        valid_lm_pred=pred_landmark[mask_lm]
        if valid_lm_gt.shape[0]>0:
            loss_l=self.loss_landmark(valid_lm_pred,valid_lm_gt)#(k,10)
            loss_l=torch.mean(loss_l)
        else:
            loss_l=torch.tensor(0.0).to(gt_landmark.device)
        total_loss=loss_c*1.0+loss_b*0.5+loss_l*0.5
        return total_loss,loss_c,loss_b,loss_l
class RNet(nn.Module):
    def __init__(self):
        super(RNet,self).__init__()
        self.flatten=nn.Flatten()
        self.backbone=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3),
                                    nn.PReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
                                    nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3),
                                    nn.PReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
                                    nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),
                                    nn.PReLU())
        self.fc=nn.Sequential(nn.Linear(64*3*3,128),
                              nn.PReLU())
        self.cls_layer=nn.Linear(128,2)
        self.bbox_layer=nn.Linear(128,4)
        self.landmark_layer=nn.Linear(128,10)
    def forward(self,x):
        x=self.backbone(x)
        x=self.flatten(x)
        x=self.fc(x)
        cls_out=self.cls_layer(x)
        bbox_out=self.bbox_layer(x)
        lm_out=self.landmark_layer(x)
        return cls_out,bbox_out,lm_out
def get_rnet_dataloader(batchsize=384):
    cls_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/rnet_cls.pkl')
    part_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/rnet_roi.pkl')
    neg_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/rnet_neg.pkl')
    full_dataset=Data.ConcatDataset([cls_data,part_data,neg_data])
    loader=Data.DataLoader(dataset=full_dataset,batch_size=batchsize,shuffle=True,num_workers=0)
    return loader
def get_onet_dataloader(batch_size=384):
    cls_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/onet_cls.pkl')
    part_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/onet_roi.pkl')
    neg_data=MTCNNDataset('D:/Code/pytorch/Design/mtcnn/data_process/data/onet_neg.pkl')
    full_dataset=Data.ConcatDataset([cls_data,part_data,neg_data])
    loader=Data.DataLoader(dataset=full_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    return loader
class ONet(nn.Module):
    def __init__(self):
        super(ONet,self).__init__()
        self.backbone=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),
                                    nn.PReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
                                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
                                    nn.PReLU(),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
                                    nn.PReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
                                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2,stride=1),
                                    nn.PReLU())
        self.flatten=nn.Flatten()
        self.fc=nn.Sequential(nn.Linear(128*3*3,256),
                              nn.PReLU())
        self.cls_layer=nn.Linear(256,2)
        self.bbox_layer=nn.Linear(256,4)
        self.landmark_layer=nn.Linear(256,10)
    def forward(self,x):
        x=self.backbone(x)
        x=self.flatten(x)
        x=self.fc(x)
        cls_out=self.cls_layer(x)
        box_out=self.bbox_layer(x)
        landmark_out=self.landmark_layer(x)
        return cls_out,box_out,landmark_out
            


