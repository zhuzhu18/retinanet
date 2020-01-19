import torch
# from retinanet import model
#
# retinanet = model.resnet18(num_classes=80, pretrained=False)
#
# retinanet.eval()
# x = torch.rand(2, 3, 224, 224)
# y = retinanet(x)

# x = torch.tensor([[1,2,3], [4,5,6]])
# y = torch.full_like(x, 5)
# print(y)

import torch

a = [torch.tensor([1.]), torch.tensor([2.]), torch.tensor([3.])]

print(torch.stack(a).mean(dim=0, keepdim=True))