import torch
from nets.resnet50 import ResNet,Bottleneck
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time


PATH = './logs/resnet50-mnist.pth'

Batch_Size = int(input('每次预测手写字体图片个数：'))
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
model.load_state_dict(torch.load(PATH))
model = model.cuda()
model.eval()
test_dataset = datasets.MNIST(root='data/', train=False,
                                    transform=transforms.ToTensor(), download=False)
gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=True)

while True:

    images, lables = next(iter(gen_test))
    img = torchvision.utils.make_grid(images, nrow=Batch_Size)
    img_array = img.numpy().transpose(1, 2, 0)

    start_time = time.time()
    outputs = model(images.cuda())
    _, id = torch.max(outputs.data, 1)
    end_time = time.time()

    print('预测用时：', end_time-start_time)
    print('预测结果为', id.data.cpu().numpy())

    cv2.imshow('img', img_array)
    cv2.waitKey(0)


