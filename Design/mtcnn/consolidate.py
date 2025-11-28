import torch
import cv2
import numpy as np
import os
import sys
sys.path.append('D:/Code/pytorch/Design/mtcnn/data_process')
sys.path.append('D:/Code/pytorch/Design/mtcnn/training')
from model_rely import PNet,RNet,ONet
from function import IoU,NMS,convert_to_square,NMS_Indices
class MTCNNDetector:
    def __init__(self):
        self.device=torch.device('cuda')
        self.pnet=PNet().to(self.device)
        self.rnet=RNet().to(self.device)
        self.onet=ONet().to(self.device)
        self.pnet.load_state_dict(torch.load('D:/Code/pytorch/Design/pnet.pth',map_location=self.device))
        self.rnet.load_state_dict(torch.load('D:/Code/pytorch/Design/rnet.pth',map_location=self.device))
        self.onet.load_state_dict(torch.load('D:/Code/pytorch/Design/onet.pth',map_location=self.device))
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
    def pnet_detect(self,image):
        boxes=[]
        img=image
        h,w,_=img.shape
        min_face_size=20
        scale=1.0
        scales=[]
        factor=0.709
        m=12.0/min_face_size
        min_side=min(h,w)*m
        cur_scale=m
        while min_side>=12:
            scales.append(cur_scale)
            cur_scale*=factor
            min_side*=factor
        for s in scales:
            _w=int(w*s)
            _h=int(h*s)
            img_resized=cv2.resize(img,(_w,_h))
            img_tensor=torch.tensor(img_resized,dtype=torch.float32)
            img_tensor=(img_tensor-127.5)/127.5
            img_tensor=img_tensor.permute(2,0,1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cls_out,box_out,_=self.pnet(img_tensor)
                cls_out=torch.softmax(cls_out,dim=1)
            cls_map=cls_out[0,1,:,:].cpu().numpy()
            off_map=box_out[0,:,:,:].cpu().numpy()
            idx=np.where(cls_map>0.6)
            if idx[0].size==0:
                continue
            stride=2
            cell_size=12
            tx1,ty1,tx2,ty2=[off_map[i,idx[0],idx[1]] for i in range(4)]
            score=cls_map[idx[0],idx[1]]
            nx1=(idx[1]*stride)/s
            ny1=(idx[0]*stride)/s
            nx2=(idx[1]*stride+cell_size)/s
            ny2=(idx[0]*stride+cell_size)/s
            w_box=nx2-nx1
            h_box=ny2-ny1
            real_x1=nx1+tx1*w_box
            real_y1=ny1+ty1*h_box
            real_x2=nx2+tx2*w_box
            real_y2=ny2+ty2*h_box
            batch_box=np.stack([real_x1,real_y1,real_x2,real_y2,score],axis=1)
            boxes.append(batch_box)
        if len(boxes)==0:
            return np.array([])
        boxes=np.vstack(boxes)
        keep=NMS(boxes,0.5)
        return keep
    def rnet_detect(self,img,boxes):
        if boxes.shape[0]==0:
            return np.array([])
        boxes=convert_to_square(boxes)
        img_24=[]
        valid_box=[]
        h,w,_=img.shape
        for box in boxes:
            nx1=int(max(0,box[0]))
            ny1=int(max(0,box[1]))
            nx2=int(min(w,box[2]))
            ny2=int(min(h,box[3]))
            crop_img=img[ny1:ny2+1,nx1:nx2+1]
            if crop_img.size==0 or crop_img.shape[0]<5 or crop_img.shape[1]<5:
                continue
            crop_resized=cv2.resize(crop_img,(24,24))
            img_24.append(crop_resized)
            valid_box.append(box)
        if len(img_24)==0:
            return np.array([])
        img_24=np.array(img_24,dtype=np.float32)
        img_tensor=torch.tensor(img_24,dtype=torch.float32)
        img_tensor=(img_tensor-127.5)/127.5
        img_tensor=img_tensor.permute(0,3,1,2)
        img_tensor=img_tensor.to(self.device)
        with torch.no_grad():
            cls_out,box_out,_=self.rnet(img_tensor)
            cls_out=torch.softmax(cls_out,dim=1)
        cls_scores=cls_out[:,1].cpu().numpy()
        box_offsets=box_out.cpu().numpy()
        keep_idxs=np.where(cls_scores>0.7)[0]
        if len(keep_idxs)==0:
            return np.array([])
        valid_box=np.array(valid_box)
        keep_boxes=valid_box[keep_idxs]
        keep_scores=cls_scores[keep_idxs]
        keep_offsets=box_offsets[keep_idxs]
        bw=keep_boxes[:,2]-keep_boxes[:,0]
        bh=keep_boxes[:,3]-keep_boxes[:,1]
        align_boxes=np.zeros_like(keep_boxes)
        align_boxes[:,0]=keep_boxes[:,0]+keep_offsets[:,0]*bw
        align_boxes[:,1]=keep_boxes[:,1]+keep_offsets[:,1]*bh
        align_boxes[:,2]=keep_boxes[:,2]+keep_offsets[:,2]*bw
        align_boxes[:,3]=keep_boxes[:,3]+keep_offsets[:,3]*bh
        align_boxes[:,4]=keep_scores
        keep=NMS(align_boxes,0.5)
        return keep
    def onet_detect(self,img,boxes):
        if boxes.shape[0]==0:
            return np.array([])
        boxes=convert_to_square(boxes)
        img_24=[]
        valid_box=[]
        h,w,_=img.shape
        for box in boxes:
            nx1=int(max(0,box[0]))
            ny1=int(max(0,box[1]))
            nx2=int(min(w,box[2]))
            ny2=int(min(h,box[3]))
            crop_img=img[ny1:ny2+1,nx1:nx2+1]
            if crop_img.size==0 or crop_img.shape[0]<5 or crop_img.shape[1]<5:
                continue
            crop_resized=cv2.resize(crop_img,(48,48))
            img_24.append(crop_resized)
            valid_box.append(box)
        if len(img_24)==0:
            return np.array([])
        img_24=np.array(img_24,dtype=np.float32)
        img_tensor=torch.tensor(img_24,dtype=torch.float32)
        img_tensor=(img_tensor-127.5)/127.5
        img_tensor=img_tensor.permute(0,3,1,2)
        img_tensor=img_tensor.to(self.device)
        with torch.no_grad():
            cls_out,box_out,lm_out=self.onet(img_tensor)
            cls_out=torch.softmax(cls_out,dim=1)
        cls_scores=cls_out[:,1].cpu().numpy()
        box_offsets=box_out.cpu().numpy()
        lm_offsets=lm_out.cpu().numpy()
        keep_idxs=np.where(cls_scores>0.7)[0]
        if len(keep_idxs)==0:
            return np.array([]),np.array([])
        valid_box=np.array(valid_box)
        keep_boxes=valid_box[keep_idxs]
        keep_scores=cls_scores[keep_idxs]
        keep_offsets=box_offsets[keep_idxs]
        keep_lms=lm_offsets[keep_idxs]
        bw=keep_boxes[:,2]-keep_boxes[:,0]
        bh=keep_boxes[:,3]-keep_boxes[:,1]
        align_boxes=np.zeros_like(keep_boxes)
        align_boxes[:,0]=keep_boxes[:,0]+keep_offsets[:,0]*bw
        align_boxes[:,1]=keep_boxes[:,1]+keep_offsets[:,1]*bh
        align_boxes[:,2]=keep_boxes[:,2]+keep_offsets[:,2]*bw
        align_boxes[:,3]=keep_boxes[:,3]+keep_offsets[:,3]*bh
        align_boxes[:,4]=keep_scores
        landmarks = np.zeros_like(keep_lms)
        for i in range(5):
            landmarks[:, 2*i] = keep_boxes[:, 0] + keep_lms[:, 2*i] * bw
            landmarks[:, 2*i+1] = keep_boxes[:, 1] + keep_lms[:, 2*i+1] * bh
        keep=NMS_Indices(align_boxes,0.6,isMin=True)#注意这里isMin的含义
        return align_boxes[keep], landmarks[keep]
    def detect_face(self,image):
        boxes=self.pnet_detect(image)
        if boxes.shape[0]==0:
            return np.array([]),np.array([])
        boxes=self.rnet_detect(image,boxes)
        if boxes.shape[0]==0:
            return np.array([]),np.array([])
        boxes,landmark=self.onet_detect(image,boxes)
        return boxes,landmark
if __name__=='__main__':
    mtcnn=MTCNNDetector()
    index=None
    for i in range(11):
        cap=cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            index=i
            print('successfully find')
            cap.release()
            break
        cap.release()
    if index==None:
        exit()
    cap=cv2.VideoCapture(index)
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret==True:
            boxes,lm=mtcnn.detect_face(frame)
            for box in boxes:
                cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
            for land in lm:
                for k in range(5):
                    cv2.circle(frame,(int(land[k*2]),int(land[k*2+1])),1,(0,0,255))
            cv2.imshow('video',frame)
        if cv2.waitKey(int(1000/30))&0xff==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





        

        