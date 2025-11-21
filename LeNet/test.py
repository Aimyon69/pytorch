import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
import torch.utils.data as Data
test_data=FashionMNIST(root='./data',train=False,transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]),download=True)
def test_data_process():
    test_data=FashionMNIST(root='./data',train=False,transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]),download=True)
    test_data_loader=Data.DataLoader(dataset=test_data,batch_size=1,shuffle=True,num_workers=0)
    return test_data_loader
def test_model_process(model,test_data_loader):
    device=torch.device('cuda')
    model=model.to(device)
    test_corrects=0
    test_num=0
    with torch.no_grad():
        for test_data_x,test_data_y in test_data_loader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            model.eval()
            output=model(test_data_x)
            pre_lab=torch.argmax(output,dim=1)
            test_corrects+=torch.sum(pre_lab==test_data_y.data)
            test_num+=test_data_x.size(0)
        test_acc=test_corrects.double().item()/test_num
        print(f'acc:{test_acc}')
if __name__=='__main__':
    model=LeNet()
    model=torch.load('best_model.pth', weights_only=False)#or:torch.serialization.add_safe_globals([LeNet])
    test_data_loader=test_data_process()
    #test_model_process(model,test_data_loader)
    device=torch.device('cuda')
    model=model.to(device)
    with torch.no_grad():
        for b_x,b_y in test_data_loader:
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.eval()
            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            result=pre_lab.item()
            label=b_y.item()
            print(f'predict:{test_data.classes[result]}  value:{test_data.classes[label]}')




