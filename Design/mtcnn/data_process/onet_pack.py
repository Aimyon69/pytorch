import pickle
import cv2
import numpy as np
import torch
import random
def pack(txt_file,save_pkl,num_limit=0):
    with open(txt_file,'r') as f:
        lines=f.readlines()
    if num_limit>0:
        if len(lines)>num_limit:
            random.shuffle(lines)
            lines=lines[:num_limit]
    data_list=[]
    for i,line in enumerate(lines):
        strs=line.strip().split()
        label=int(strs[1])
        roi=list(map(float,strs[2:6]))
        pts=list(map(float,strs[6:]))
        img=cv2.imread(strs[0])
        if img is None:
            continue
        img_tensor=torch.tensor(img,dtype=torch.float32)
        img_tensor=(img_tensor-127.5)/127.5
        img_tensor=img_tensor.permute(2,0,1)
        data_item={
            'image':img_tensor,
            'label':torch.tensor(label,dtype=torch.long),
            'roi':torch.tensor(roi,dtype=torch.float32),
            'pts':torch.tensor(pts,dtype=torch.float32)
        }
        data_list.append(data_item)
    with open(save_pkl,'wb') as f:
        pickle.dump(data_list,f)
    print('end')
    return len(data_list)
if __name__=='__main__':
    pos_num=pack('D:/Code/pytorch/Design/mtcnn/data_process/data/pos_48.txt','D:/Code/pytorch/Design/mtcnn/data_process/data/onet_cls.pkl')
    print(pos_num)
    pack('D:/Code/pytorch/Design/mtcnn/data_process/data/part_48.txt','D:/Code/pytorch/Design/mtcnn/data_process/data/onet_roi.pkl',pos_num)
    pack('D:/Code/pytorch/Design/mtcnn/data_process/data/neg_48.txt','D:/Code/pytorch/Design/mtcnn/data_process/data/onet_neg.pkl',3*pos_num)

