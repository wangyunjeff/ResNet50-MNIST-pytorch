import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


class Bottleneck(nn.Module):
    '''
    包含三种卷积层
    conv1-压缩通道数
    conv2-提取特征
    conv3-扩展通道数
    这种结构可以更好的提取特征，加深网络，并且可以减少网络的参数量。
    '''

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        这块实现了残差块结构

        ResNet50有两个基本的块，分别名为Conv Block和Identity Block，renet50就是利用了这两个结构堆叠起来的。
        它们最大的差距是残差边上是否有卷积。

        Identity Block是正常的残差结构，其残差边没有卷积，输入直接与输出相加；
        Conv Block的残差边加入了卷积操作和BN操作（批量归一化），其作用是可以通过改变卷积操作的步长通道数，达到改变网络维度的效果。

        也就是说
        Identity Block输入维度和输出维度相同，可以串联，用于加深网络的；
        Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度。
        :param
        x:输入数据
        :return:
        out:网络输出结果
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        用于构造Conv Block 和 Identity Block的堆叠
        :param block:就是上面的Bottleneck，用于实现resnet50中最基本的残差块结构
        :param planes:输出通道数
        :param blocks:残差块重复次数
        :param stride:步长
        :return:
        构造好的Conv Block 和 Identity Block的堆叠网络结构
        '''
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#

        # 边（do构建Conv Block的残差wnsample）
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [] # 用于堆叠Conv Block 和 Identity Block
        # 添加一层Conv Block
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 添加完后输入维度变了，因此改变inplanes（输入维度）
        self.inplanes = planes * block.expansion
        # 添加blocks层 Identity Block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier
