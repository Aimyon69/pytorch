from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
train_data=FashionMNIST(root='./data',train=True,transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),download=True)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
for step,(b_x,b_y) in enumerate(train_loader):
    if step >0:
        break
    batch_x=b_x.squeeze().numpy()
    batch_y=b_y.numpy()
    class_label=train_data.classes
    print(class_label)
for i in range(len(batch_y)):
    plt.subplot(4,16,i+1)
    plt.imshow(batch_x[i,:,:],'gray')
    plt.title(class_label[batch_y[i]],size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
