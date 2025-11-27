import sys
import os
import cv2 
import numpy as np
import torch
from function import IoU,convert_to_square,NMS
sys.path.append('D:/Code/pytorch/Design/mtcnn/training')
from model_rely import PNet

image_size = 24
batch_size = 384
pnet_weight = 'D:/Code/pytorch/Design/pnet.pth'
anno_file = 'D:/Code/pytorch/Design/celebA/list_bbox_celeba.txt'
landmark_file = 'D:/Code/pytorch/Design/celebA/list_landmarks_celeba.txt' 
im_dir = 'D:/Code/pytorch/Design/celebA/img_celeba'
save_dir = 'D:/Code/pytorch/Design/mtcnn/data_process/data'

def pnet_detect(model,image,device):
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
        img_tensor=img_tensor.permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_out,box_out,_=model(img_tensor)
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
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PNet()
    model.load_state_dict(torch.load(pnet_weight, map_location=device))
    model=model.to(device)
    model.eval()
    if not os.path.exists(os.path.join(save_dir, 'img_rnet/pos_image')):
        os.makedirs(os.path.join(save_dir, 'img_rnet/pos_image'))
    if not os.path.exists(os.path.join(save_dir, 'img_rnet/part_image')):
        os.makedirs(os.path.join(save_dir, 'img_rnet/part_image'))
    if not os.path.exists(os.path.join(save_dir, 'img_rnet/neg_image')):
        os.makedirs(os.path.join(save_dir, 'img_rnet/neg_image'))

    f_pos=open(save_dir+'/pos_24.txt','w')
    f_part=open(save_dir+'/part_24.txt','w')
    f_neg=open(save_dir+'/neg_24.txt','w')
    
    with open(anno_file,'r') as f:
        annotations=f.readlines()
    with open(landmark_file, 'r') as f_l:
        landmark_annotations = f_l.readlines()

    idx=0
    for i,line in enumerate(annotations[2:]):
        strs=line.strip().split()
        img_path=os.path.join(im_dir,strs[0])
        x1,y1,w,h=map(float,strs[1:])
        x2,y2=x1+w-1,y1+h-1
        gt_box=np.array([x1,y1,x2,y2])
        lm_line = landmark_annotations[i+2].strip().split()
        landmark_gt = list(map(float, lm_line[1:]))

        img=cv2.imread(img_path)
        if img is None: continue 

        boxes=pnet_detect(model,img,device)
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
            if wid<10 or hei<10 or wid!=hei:
                continue
            
            iou_val=IoU(np.array([nx1,ny1,nx2,ny2]),gt_box.reshape(1,-1))
            
            crop_img=img[ny1:ny2+1,nx1:nx2+1]
            if crop_img.size == 0: continue 
            resized_img=cv2.resize(crop_img,(image_size,image_size))
            
            off_x1=(x1-nx1)/wid
            off_y1=(y1-ny1)/hei
            off_x2=(x2-nx2)/wid
            off_y2=(y2-ny2)/hei
            valid_pts = True
            for k in range(5):
                pt_x = landmark_gt[2*k]
                pt_y = landmark_gt[2*k+1]
                if not (nx1 <= pt_x <= nx2 and ny1 <= pt_y <= ny2):
                    valid_pts = False
                    break
            if valid_pts:
                pts_offsets = []
                for k in range(5):
                    pts_offsets.append((landmark_gt[2*k] - nx1) / wid)
                    pts_offsets.append((landmark_gt[2*k+1] - ny1) / hei)
                pts_str = " ".join([f"{p:.3f}" for p in pts_offsets])
            else:
                pts_str = "0 0 0 0 0 0 0 0 0 0"
            zeros_str = "0 0 0 0 0 0 0 0 0 0"
            if iou_val>=0.65:
                save_img=save_dir+f'/img_rnet/pos_image/{idx}.jpg'
                write_pts = pts_str if valid_pts else zeros_str
                
                f_pos.write(f'{save_img} 1 {off_x1} {off_y1} {off_x2} {off_y2} {write_pts}\n')
                cv2.imwrite(save_img,resized_img)
                idx+=1
            elif 0.4<=iou_val<=0.65:
                save_img=save_dir+f'/img_rnet/part_image/{idx}.jpg'
                f_part.write(f'{save_img} -1 {off_x1} {off_y1} {off_x2} {off_y2} {zeros_str}\n')
                cv2.imwrite(save_img,resized_img)
                idx+=1
            elif iou_val<0.3:
                save_img=save_dir+f'/img_rnet/neg_image/{idx}.jpg'
                f_neg.write(f'{save_img} 0 0 0 0 0 {zeros_str}\n')
                cv2.imwrite(save_img,resized_img)
                idx+=1
        if i % 100 == 0:
            print(f"Processing {i}, samples generated: {idx}")

    f_pos.close()
    f_part.close()
    f_neg.close()
    print('end')

if __name__=='__main__':
    main()

        



