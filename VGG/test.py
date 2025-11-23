import torch
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
from torchvision import transforms
from model import VGG
test_data=FashionMNIST(root='D:/Code/pytorch/LeNet/data',train=False,transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),download=True)
def test_data_process():
    test_data_loader=Data.DataLoader(dataset=test_data,batch_size=1,num_workers=0)
    return test_data_loader
def test_model_process(model,test_data_loader):
    device=torch.device('cuda')
    model=torch.load('best_model.pth',weights_only=False)
    model=model.to(device)
    test_corrects=0
    test_num=0
    model.eval()
    with torch.no_grad():
        for d_x,d_y in test_data_loader:
            d_x=d_x.to(device,non_blocking=True)
            d_y=d_y.to(device,non_blocking=True)
            output=model(d_x)
            pre_lab=torch.argmax(output,dim=1)
            test_corrects+=torch.sum(pre_lab==d_y.data).item()
            test_num+=d_x.size(0)
            print(f'predict:{test_data.classes[pre_lab.item()]} value:{test_data.classes[d_y.item()]}')
        print(f'test acc:{test_corrects/test_num}')
if __name__=='__main__':
    test_data_loader=test_data_process()
    model=VGG()
    test_model_process(model,test_data_loader)

