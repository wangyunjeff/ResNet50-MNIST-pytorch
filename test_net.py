import torch
from torchsummary import summary

# from nets.CSPdarknet import darknet53
# from nets.yolo4 import YoloBody
from nets.resnet50 import ResNet,Bottleneck
from torch import sum
if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(1, 28, 28))

pass