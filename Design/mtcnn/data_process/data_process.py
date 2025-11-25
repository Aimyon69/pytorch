from function import IoU
import os
import numpy as np
import cv2
import torch
import random
from torch.utils.data import Dataset
import pickle
image_size=48
anno_file='D:/Code/pytorch/Design/celebA/list_bbox_celeba.txt'
landmark_file='D:/Code/pytorch/Design/celebA/list_landmarks_celeba.txt'
im_dir='D:/Code/pytorch/Design/celebA/img_celeba'
save_dir='./data'
#分别读取人脸框标注数据和特征点标注数据
os.makedirs(save_dir, exist_ok=True)
with open(anno_file, 'r') as f:
    annotations = f.readlines()
with open(landmark_file, 'r') as f2:
    landmark_positions_list = f2.readlines()
pos_cls_list=[]#    
pos_roi_list=[]#   
neg_cls_list=[]#    
part_roi_list=[]#   
pts_list=[]#        
p_idx=0#  
n_idx=0# 
d_idx=0#  
pts_idx=0#  
for id_annos in range(2, 10000):
    #从2开始是为了跳过第0行(样本数量)，第1行(各项名称)
    anno = annotations[id_annos].strip().split()#去除前后空格和根据空格分割元素存储到列表中
    im_path = anno[0]#第一个元素是图片文件名
    x1, y1, w, h = map(float, anno[1:])
    x2 = x1 + w - 1  
    y2 = y1 + h - 1
    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    landmark_anno = landmark_positions_list[id_annos].strip().split()
    pts_raw = list(map(float, landmark_anno[1:]))#读取五个特征点数据，每个数据有x，y两个属性，list共10个元素
    img_path = os.path.join(im_dir, f"{os.path.splitext(im_path)[0]}.jpg")#splitext(im_path)[0]指的是类似image.png中的image文件名，[1]是拓展名.png
    img = cv2.imread(img_path)
    if img is None:
        continue  
    height, width, _ = img.shape#_表示忽略
    if max(w, h) < 40 or x1 < 0 or y1 < 0:
        continue#忽略较小的图像
    for _ in range(20):  
        size = int(min(w, h))  
        nx1 = max(int(x1), 0)  
        ny1 = max(int(y1), 0)  
        nx2 = nx1 + size
        ny2 = ny1 + size
        if nx2 > width or ny2 > height:
            continue
        crop_box = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
        offset_x1 = (x1 - nx1) / size
        offset_y1 = (y1 - ny1) / size
        offset_x2 = (x2 - nx2) / size
        offset_y2 = (y2 - ny2) / size
        cropped_im = img[ny1:ny2, nx1:nx2, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        iou = IoU(crop_box, box.reshape(1, -1))[0]
        if iou >= 0.65:
            im_tensor = torch.from_numpy(resized_im).permute(2, 0, 1).float() 
            im_tensor = (im_tensor - 127.5) / 127.5  
            pos_cls_list.append({
                "image": im_tensor,
                "label": torch.tensor(1, dtype=torch.long),
                "roi": torch.tensor([-1, -1, -1, -1], dtype=torch.float32) 
            })
            pos_roi_list.append({
                "image": im_tensor,
                "label": torch.tensor(1, dtype=torch.long),
                "roi": torch.tensor([offset_x1, offset_y1, offset_x2, offset_y2], dtype=torch.float32)
            })
            px_list = pts_raw[::2]  
            py_list = pts_raw[1::2] 
            if (min(px_list) > nx1 and min(py_list) > ny1 and
                max(px_list) < nx2 and max(py_list) < ny2):
                pts_x = [(px - nx1) / size for px in px_list]
                pts_y = [(py - ny1) / size for py in py_list]
                pts = pts_x + pts_y 

                pts_list.append({
                    "image": im_tensor,
                    "label": torch.tensor(1, dtype=torch.long),
                    "roi": torch.tensor([offset_x1, offset_y1, offset_x2, offset_y2], dtype=torch.float32),
                    "pts": torch.tensor(pts, dtype=torch.float32)
                })
                pts_idx += 1
            p_idx += 1
        elif iou >= 0.4:
            im_tensor = torch.from_numpy(resized_im).permute(2, 0, 1).float()
            im_tensor = (im_tensor - 127.5) / 127.5

            part_roi_list.append({
                "image": im_tensor,
                "label": torch.tensor(-1, dtype=torch.long), 
                "roi": torch.tensor([offset_x1, offset_y1, offset_x2, offset_y2], dtype=torch.float32)
            })
            d_idx += 1
    neg_num = 0
    while neg_num < 10: 
        if min(width, height) // 2 > 40:
            size = random.randint(40, min(width, height) // 2)
        else:
            size = random.randint(min(width, height) // 2, 40)
        nx = random.randint(0, width - size)
        ny = random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size], dtype=np.float32)
        iou = IoU(crop_box, box.reshape(1, -1))[0]
        if iou < 0.3:
            cropped_im = img[ny:ny+size, nx:nx+size, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            im_tensor = torch.from_numpy(resized_im).permute(2, 0, 1).float()
            im_tensor = (im_tensor - 127.5) / 127.5

            neg_cls_list.append({
                "image": im_tensor,
                "label": torch.tensor(0, dtype=torch.long),  
                "roi": torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
            })
            n_idx += 1
            neg_num += 1
    if id_annos % 100 == 0:
        print(f"处理进度: {id_annos}/10000, 关键点样本: {pts_idx}, 正样本: {p_idx}, 部分样本: {d_idx}, 负样本: {n_idx}")
if len(part_roi_list) > p_idx:
    part_keep = random.sample(range(len(part_roi_list)), p_idx)
    part_roi_list = [part_roi_list[i] for i in part_keep]
if len(neg_cls_list) > 3 * p_idx:
    neg_keep = random.sample(range(len(neg_cls_list)), 3 * p_idx)
    neg_cls_list = [neg_cls_list[i] for i in neg_keep]
cls_list = neg_cls_list + pos_cls_list 
roi_list = part_roi_list + pos_roi_list 
with open(os.path.join(save_dir, "cls_imdb.pkl"), "wb") as f:
    pickle.dump(cls_list, f)
with open(os.path.join(save_dir, "roi_imdb.pkl"), "wb") as f:
    pickle.dump(roi_list, f)
with open(os.path.join(save_dir, "pts_imdb.pkl"), "wb") as f:
    pickle.dump(pts_list, f)
print(f"数据集生成完成，保存路径: {save_dir}")
print(f"分类样本数: {len(cls_list)}, 回归样本数: {len(roi_list)}, 关键点样本数: {len(pts_list)}")
