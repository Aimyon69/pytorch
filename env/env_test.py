import numpy
import pandas
import torch
flag=torch.cuda.is_available()
print(flag)
print(torch.cuda.get_arch_list())
device=torch.device('cuda:0')
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
