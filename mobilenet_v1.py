import torch.nn as nn
import torch
class Conv_bn(nn.Module):
    def __init__(self,inp,oup,stride):
        super(Conv_bn, self).__init__()
        self.convBn=nn.Sequential(
            nn.Conv2d(inp,oup,3,stride,1,bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out=self.convBn(x)
        return out

class Conv_depth(nn.Module):
    def __init__(self,inp,oup,stride):
        super(Conv_depth, self).__init__()
        self.convDepthwise=nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out=self.convDepthwise(x)
        return out


class MobileNet(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet, self).__init__()
        self.upchannel=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1)
        self.unlinear=nn.ReLU(inplace=True)
        self.mobelnet=nn.Sequential(
            Conv_bn(3, 32, 2),
            Conv_depth(32, 64, 1),
            Conv_depth(64, 128, 2),
            Conv_depth(128, 128, 1),
            Conv_depth(128, 256, 2),
            Conv_depth(256, 256, 1),
            Conv_depth(256, 512, 2),
            Conv_depth(512, 512, 1),
            Conv_depth(512, 512, 1),
            Conv_depth(512, 512, 1),
            Conv_depth(512, 512, 1),
            Conv_depth(512, 512, 1),
            Conv_depth(512, 1024, 2),
            Conv_depth(1024, 1024, 1),
            nn.AvgPool2d(2),
            )

        self.fc = nn.Linear(2048,num_classes)

    # 网络的前向过程
    def forward(self, x):
        x=self.upchannel(x)
        x=self.unlinear(x)
        x=self.mobelnet(x)
        # print(x.shape)
        x=x.view(-1, 2048)
        x=self.fc(x)
        return x

model=MobileNet(35)

