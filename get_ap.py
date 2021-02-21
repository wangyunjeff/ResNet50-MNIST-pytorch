import torch
from nets.resnet50 import ResNet,Bottleneck
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

Batch_size = 128

root = '.\logs'
file_dir = os.listdir(root)
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)

for file in file_dir:
    PATH = os.path.join(root, file)



    model.load_state_dict(torch.load(PATH))
    model = model.cuda()
    model.eval()

    test_dataset = datasets.MNIST(root='data/', train=False,
                                      transform=transforms.ToTensor(), download=False)


    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=True)

    test_correct = 0
    for data in gen_test:
        inputs, lables = data
        inputs, lables = Variable(inputs).cuda(), Variable(lables).cuda()
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(file)
    print("correct:%.3f%%" % (100 * test_correct / len(test_dataset)))
