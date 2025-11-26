import pickle
import torch
from torch.utils.data import Dataset
import torch.utils.data  as Data
class MTCNNDataset(Dataset):
    def __init__(self,path):
        with open(path,'rb') as f:
            self.data=pickle.load(f)#返回列表，元素是字典
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample=self.data[index]
        return sample['image'],sample['label'],sample['roi'],sample.get('pts',torch.zeros(10))
if __name__=='__main__':
    cls_path='D:/Code/pytorch/Design/mtcnn/data_process/data/cls_imdb.pkl'
    roi_path='D:/Code/pytorch/Design/mtcnn/data_process/data/roi_imdb.pkl'
    pts_path='D:/Code/pytorch/Design/mtcnn/data_process/data/pts_imdb.pkl'
    cls_dataset=MTCNNDataset(cls_path)
    roi_dataset=MTCNNDataset(roi_path)
    pts_dataset=MTCNNDataset(pts_path)
    print('读取数据集成功，各数据集的长度为:')
    print(f'cls_dataset:{len(cls_dataset)}')
    print(f'roi_dataset:{len(roi_dataset)}')
    print(f'pts_dataset:{len(pts_dataset)}')
    print('-'*20)
    cls_dataset_loader=Data.DataLoader(dataset=cls_dataset,shuffle=True,batch_size=10,num_workers=0)
    roi_dataset_loader=Data.DataLoader(dataset=roi_dataset,shuffle=True,batch_size=10,num_workers=0)
    pts_dataset_loader=Data.DataLoader(dataset=pts_dataset,shuffle=True,batch_size=10,num_workers=0)
    for step,(image,label,roi,pts) in enumerate(cls_dataset_loader):
        if step>0:
            break
        print(label,roi,pts,sep="***")

