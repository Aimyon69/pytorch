import torch
import os
import cv2
import numpy as np
from function import NMS,convert_to_square,IoU
import sys
sys.path.append('D:/Code/pytorch/Design/mtcnn/training')
from model_rely import PNet,RNet
from rnet_data_process import pnet_detect
image_size = 48
batch_size = 384
pnet_weight = 'D:/Code/pytorch/Design/pnet.pth'
rnet_weight = 'D:/Code/pytorch/Design/rnet.pth'
anno_file = 'D:/Code/pytorch/Design/celebA/list_bbox_celeba.txt'
landmark_file = 'D:/Code/pytorch/Design/celebA/list_landmarks_celeba.txt' 
im_dir = 'D:/Code/pytorch/Design/celebA/img_celeba'
def rnet_detect(model,img,boxes,device):
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
    img_tensor=img_tensor.to(device)
    with torch.no_grad():
        cls_out,box_out,_=model(img_tensor)
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
def main():
    device=torch.device('cuda')
    pnet=PNet().to(device)
    rnet=RNet().to(device)
    pnet.load_state_dict(torch.load('D:/Code/pytorch/Design/pnet.pth'))
    rnet.load_state_dict(torch.load('D:/Code/pytorch/Design/rnet.pth'))
    pnet.eval()
    rnet.eval()
    f_pos=open('D:/Code/pytorch/Design/mtcnn/data_process/data/pos_48.txt','w')
    f_part=open('D:/Code/pytorch/Design/mtcnn/data_process/data/part_48.txt','w')
    f_neg=open('D:/Code/pytorch/Design/mtcnn/data_process/data/neg_48.txt','w')
    with open(anno_file,'r') as f:
        annotations=f.readlines()
    with open(landmark_file,'r') as f:
        landmark_annos=f.readlines()
    idx=0
    for i,line in enumerate(annotations[2:]):
        strs=line.strip().split()
        img_path=os.path.join(im_dir,strs[0])
        x1,y1,w,h=map(float,strs[1:])
        gt_box=np.array([x1,y1,x1+w-1,y1+h-1])
        lm_strs=landmark_annos[i+2].strip().split()
        landmark_gt=list(map(float,lm_strs[1:]))
        img=cv2.imread(img_path)
        if img is None:
            continue
        boxes=pnet_detect(pnet,img,device)
        if boxes.shape[0]==0:
            continue
        boxes=rnet_detect(rnet,img,boxes,device)
        if boxes.shape[0]==0:
            continue
        boxes=convert_to_square(boxes)
        for box in boxes:
            nx1,ny1,nx2,ny2,_=box
            nx1=int(max(0,nx1))
            ny1=int(max(0,ny1))
            nx2=int(min(img.shape[1],nx2))
            ny2=int(min(img.shape[0],ny2))
            wid=nx2-nx1
            hei=ny2-ny1
            if wid<20 or hei<20:
                continue
            iou_val=IoU(np.array([nx1,ny1,nx2,ny2]),gt_box.reshape(1,-1))
            crop_img=img[ny1:ny2+1,nx1:nx2+1]
            resized_img=cv2.resize(crop_img,(image_size,image_size))
            off_x1=(gt_box[0]-nx1)/wid
            off_y1=(gt_box[1]-ny1)/hei
            off_x2=(gt_box[2]-nx2)/wid
            off_y2=(gt_box[3]-ny2)/hei
            valid_pts=True
            for k in range(5):
                if not(nx1<=landmark_gt[2*k]<=nx2 and ny1 <=landmark_gt[2*k+1]<=ny2):
                    valid_pts=False
                    break
            pts_str='0 '*10
            if valid_pts:
                pts_offsets=[]
                for k in range(5):
                    pts_offsets.append((landmark_gt[2*k]-nx1)/wid)
                    pts_offsets.append((landmark_gt[2*k+1]-ny1)/hei)
                pts_str=' '.join([f"{p:.3f}" for p in pts_offsets])
            if iou_val>=0.65:
                if valid_pts:
                    path='D:/Code/pytorch/Design/mtcnn/data_process/data/img_onet/pos_image'
                    f_pos.write(f'{path}/{idx}.jpg 1 {off_x1} {off_y1} {off_x2} {off_y2} {pts_str}\n')
                    cv2.imwrite(path+f'/{idx}.jpg',resized_img)
                else:
                    path='D:/Code/pytorch/Design/mtcnn/data_process/data/img_onet/pos_image'
                    f_pos.write(f'{path}/{idx}.jpg 1 {off_x1} {off_y1} {off_x2} {off_y2} 0 0 0 0 0 0 0 0 0 0\n')
                    cv2.imwrite(path+f'/{idx}.jpg',resized_img)
                idx+=1
            elif 0.4<=iou_val<0.65:
                path='D:/Code/pytorch/Design/mtcnn/data_process/data/img_onet/part_image'
                f_part.write(f'{path}/{idx}.jpg -1 {off_x1} {off_y1} {off_x2} {off_y2} 0 0 0 0 0 0 0 0 0 0\n')
                cv2.imwrite(path+f'/{idx}.jpg',resized_img)
                idx+=1
            elif iou_val<0.3:
                path='D:/Code/pytorch/Design/mtcnn/data_process/data/img_onet/neg_image'
                f_neg.write(f'{path}/{idx}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n')
                cv2.imwrite(path+f'/{idx}.jpg',resized_img)
                idx+=1
        if i%100==0:
            print(f"Processing {i}, generated samples: {idx}")
    f_pos.close()
    f_part.close()
    f_neg.close()
    print('end')
if __name__=='__main__':
    main()










