# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/clovaai/CutMix-PyTorch/blob/master/resnet.py

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    # model_name, args.depth, args.num_classes, args.bottleneck, args.prete, args.resolution
    def __init__(self, dataset, depth, num_classes, bottleneck=False, pretext='None', resolution=224):
        super(ResNet, self).__init__()        
        self.dataset = dataset
        self.fc2 = None
        blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(resolution//32)
        self.fc_in_dim = 512 * blocks[depth].expansion
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        # FC2 is for pretext task
        self.fc2 = None
        if pretext == 'lorotI':
            self.fc2 = nn.Linear(512 * blocks[depth].expansion, 4)
        if pretext == 'lorotE':
            self.fc2 = nn.Linear(512 * blocks[depth].expansion, 16)

        # head / head2 for supervised contrastive learning
        self.head = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion), nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                  nn.Linear(512 * blocks[depth].expansion, 1024))
        self.head2 = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion),
                                  nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                  nn.Linear(512 * blocks[depth].expansion, 1024))

        # head_fc for balanced contrastive learning (BCL)
        self.head_fc = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion),
                                   nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                   nn.Linear(512 * blocks[depth].expansion, 1024))



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, nce=False, val=False, BCL=False, fgssl=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)

        if fgssl:
            return x1, x2, x3, x4, x5
        if nce: ## Supcon
            if val:
                return x
            if self.fc2 is not None:
                return F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), self.fc2(x)
            return F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), None
        x1 = self.fc(x)
        if BCL: ## Balanced Contrastive Learning
            centers_logits = F.normalize(self.head_fc(self.fc.weight), dim=1)

            if self.fc2 is None:
                return x1, None, F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), centers_logits
            x2 = self.fc2(x)
            return x1, x2, F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), centers_logits

        if self.fc2 is not None:
            x2 = self.fc2(x)
            return x1, x2
        return x1, None
    

class ResNet_patch16(nn.Module):
    # model_name, args.depth, args.num_classes, args.bottleneck, args.prete
    def __init__(self, dataset, depth, num_classes, bottleneck=False, pretext='None', patch_num=16):
        super(ResNet_patch16, self).__init__()
        self.patch_num = patch_num 
        self.dataset = dataset
        self.fc2 = None
        blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.inplanes = 64       
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_patch = nn.Conv2d(3*self.patch_num, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():    
            for i in range(patch_num):
                start_idx = i * 3
                end_idx = start_idx + 3
                self.conv1_patch.weight[:, start_idx:end_idx, :, :] = self.conv1.weight
                
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_in_dim = 512 * blocks[depth].expansion
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        # FC2 is for pretext task
        self.fc2 = None
        if pretext == 'lorotI':
            self.fc2 = nn.Linear(512 * blocks[depth].expansion, 4)
        if pretext == 'lorotE':
            self.fc2 = nn.Linear(512 * blocks[depth].expansion, 16)

        # head / head2 for supervised contrastive learning
        self.head = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion), nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                  nn.Linear(512 * blocks[depth].expansion, 1024))
        self.head2 = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion),
                                  nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                  nn.Linear(512 * blocks[depth].expansion, 1024))

        # head_fc for balanced contrastive learning (BCL)
        self.head_fc = nn.Sequential(nn.Linear(512 * blocks[depth].expansion, 512 * blocks[depth].expansion),
                                   nn.BatchNorm1d(512 * blocks[depth].expansion), nn.ReLU(inplace=True),
                                   nn.Linear(512 * blocks[depth].expansion, 1024))



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, nce=False, val=False, BCL=False, fgssl=False):   
        B, _, _, _ = x.shape
      
        reshaped = x.unfold(2, 224, 224).unfold(3, 224, 224)

        # Step 2: Rearrange dimensions to combine 4x4 blocks into a single dimension
        # Permute dimensions to: [64, 4, 4, 3, 224, 224]
        reshaped = reshaped.permute(0, 2, 3, 1, 4, 5)

        # Step 3: Combine the 4x4 blocks into a single dimension
        # Final shape: [64, 48, 224, 224]
        x = reshaped.reshape(B, -1, 224, 224)      
        x = self.conv1_patch(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)

        if fgssl:
            return x1, x2, x3, x4, x5
        if nce: ## Supcon
            if val:
                return x
            if self.fc2 is not None:
                return F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), self.fc2(x)
            return F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), None
        x1 = self.fc(x)
        if BCL: ## Balanced Contrastive Learning
            centers_logits = F.normalize(self.head_fc(self.fc.weight), dim=1)

            if self.fc2 is None:
                return x1, None, F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), centers_logits
            x2 = self.fc2(x)
            return x1, x2, F.normalize(self.head(x), dim=1), F.normalize(self.head2(x), dim=1), centers_logits

        if self.fc2 is not None:
            x2 = self.fc2(x)
            return x1, x2
        return x1, None