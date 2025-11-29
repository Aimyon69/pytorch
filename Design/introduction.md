# ***基于MTCNN~(人脸检测)~、ResNet~(卷积神经网络)~与OpenCV~(图像处理)~构建的人脸情绪识别模型***
---
## 获取模型源代码可以访问我的GitHub仓库：[Aimyon69-Design](https://github.com/Aimyon69/pytorch/tree/master/Design)
---
## MTCNN人脸检测模块：
要实现人脸的情绪识别，首先的任务就是将视频流或者图片流中的人脸信息给提取出来，忽略其他例如背景，杂物等无关信息。在此项目中，我们选取了==MTCNN==，实现了对人脸的检测，框选和脸部特征点的提取。实现的具体细节如下所示：
### 1.数据集的选取：
在本项目中训练数据集我们使用了香港中文大学提供的一个开源大规模人脸属性数据集CelebA作为我们MTCNN网络的训练数据集。以下是数据集的官方网站：[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。因为其具有人脸检测边框数据标注和特征点坐标的标注数据，作为我们MTCNN网络的训练数据集是合理且适配的。
### 2.MTCNN的基本结构和原理：
MTCNN具有三个网络：==PNet==，==RNet==，==ONet==。
#### PNet：
PNet是一个全卷积网络，它的主要作用是粗略给出图像中人脸的预选框位置，以及特征点坐标。由于它的感受野（卷积核大小）只有12x12，所以对于图像中的大人脸是无法进行识别的。为解决此难题，引入了==图像金字塔==的方法，图像金字塔就是对输入原图像进行多次尺寸缩小，得到许多的子图像，直到子图像的宽或者高小于12个像素就停止缩小。目的就是让原图像中的无法被PNet识别的大人脸缩小到可以被PNet识别，并初步筛选出来。但是上述的图像金字塔会带来另一个问题：同一张脸被多次识别框选。原因是通过图像金字塔得到的子图中会重复出现同一张人脸，经过PNet识别，会重复的识别并框选同一张人脸。为了解决这个问题，我们引入==NMS（非极大值抑制）==，我们将高度重合的框经行一个取极大值的效果，只保留一个是人脸概率最高的框。在这个过程中当然会有不是人脸的部分被PNet识别并框选，接下来我们就需要将PNet的识别结果送给RNet进行进一步筛选。这里在阐释一下为什么PNet网络的识别精度低：==1==网络结构简单，是一个全卷积网络，没有全连接层来精确拟合数据特征。==2==PNet的卷积核的感受野较小（12x12），不能很好的提取人脸的所有特征。==3==为了解决感受野小的问题，图像金字塔的引入，导致人脸像素尺寸被强制的缩小，在此过程中人脸的许多特征细节会被丢失。
##### PNet的pytorch框架搭建：
```python
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
```
##### PNet的训练数据集准备：
简单介绍下如何从CelebA得来PNet的训练数据集的。
首先需要阐释的是为什么不能拿celebA的数据集直接来训练我们的MTCNN网络。核心原因是数据标注和类型不符合MTCNN的训练目的。我们MTCNN是一个多任务训练网络，包含==人脸分类==，==人脸框回归==，==人脸特征点回归==三个任务。对于分类任务，我们需要正样本和负样本同时喂给网络训练。对于人脸框和特征点回归任务，我们需要让网络对一个输入随机的图像，推测出人脸真实框的位置，给出的是真实框相对于现在输入的图像框的偏移量。但是CelebA只有正样本，这显然不符合我们分类任务的数据属性需要，对于回归任务来说，不满足输入随机的特性，我们需要在CelebA数据上随机裁剪图像，来生成适合训练的数据集，包含（正样本，负样本，部分样本）。
接着简单解释下如何得到各样本的：我们对图像进行随机裁剪（具体的裁剪逻辑见pnet数据处理py文件），得到许多的待分类图像，按照以下的规则进行分类：
正样本：iou值大于0.65的当作正样本，0.4-0.65的当作部分样本，0.3以下的当作负样本，具体代码如下：
```python
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
                pts = []
                for i in range(5):
                    pts.append(pts_x[i])
                    pts.append(pts_y[i]) 
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
        size = min(size, width, height)
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
```
##### 损失函数的定义：
在模型训练之前，我们介绍下MTCNN损失函数的定义，由于是多任务网络，所以我们需要对损失函数进行一个细化:
```python
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
```
总结来说，就是正，负样本的label值用来计算我们的分类loss值，正，部分样本的roi值来计算我们的人脸框回归loss值，pts值不为全0的样本用来计算我们的特征点loss值。最后按照`total_loss=loss_c*1.0+loss_b*0.5+loss_l*0.5`得到加权loss值以来反向传播计算梯度，更新参数。这里需注意以下问题：total_loss来反向传播的,但是是如何确保我的三个子损失值是按梯度下降的方向变化的。
##### PNet模型的训练：
整体框架如下：
```python
def train():
    device=torch.device('cuda')
    batch_size=384
    lr=0.001
    epochs=30
    train_loader=get_dataloader(batch_size)
    model=PNet()
    model=model.to(device)
    criterion=MTCNNLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        since=time.time()
        for step,(img,label,roi,pts) in enumerate(train_loader):
            img=img.to(device)
            label=label.to(device)
            roi=roi.to(device)
            pts=pts.to(device)
            cls_out,box_out,landmark_out=model(img)
            loss,l_cls,l_box,l_pts=criterion(cls_out,box_out,landmark_out,label,roi,pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f'epoch:{epoch}')
        print(f'loss:{epoch_loss/len(train_loader):.4f}')
        print(f'consuming time:{time.time()-since}')
        print('.'*20)
        scheduler.step()
    torch.save(model.state_dict(),'D:/Code/pytorch/Design/pnet.pth')
if __name__=='__main__':
    train()
```
训练结果如下：
![PNet训练结果](https://github.com/Aimyon69/pytorch/blob/master/Design/image/pnet_training.png)









