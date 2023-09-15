import torch
import torch.nn as nn


class lbdNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lbdNet, self).__init__()

        norm_ch = 64
        self.conv = nn.Conv2d(in_ch, norm_ch, (3, 3), 1, (1, 1), bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder1 = EncoderBlock(norm_ch, norm_ch)
        self.encoder2 = EncoderBlock(norm_ch, norm_ch)
        self.encoder3 = EncoderBlock(norm_ch, norm_ch)
        self.encoder4 = EncoderBlock(norm_ch, norm_ch)

        self.bottleneck = BasicBlock(norm_ch, norm_ch)

        self.decoder4 = FIFDecoderBlock(norm_ch * 2, norm_ch)
        self.decoder3 = FIFDecoderBlock(norm_ch * 2, norm_ch)
        self.decoder2 = FIFDecoderBlock(norm_ch * 2, norm_ch)
        self.decoder1 = FIFDecoderBlock(norm_ch * 2, norm_ch)

        self.classifier1 = ClassifierBlock(norm_ch, out_ch)
        self.classifier2 = ClassifierBlock(norm_ch, out_ch - 1)

    def forward(self, input):
        tmp = self.conv(input)
        e1, out1, idx1 = self.encoder1(tmp)
        e2, out2, idx2 = self.encoder2(e1)
        e3, out3, idx3 = self.encoder3(e2)
        e4, out4, idx4 = self.encoder4(e3)

        bn = self.bottleneck(e4)

        s4, b4 = self.decoder4(bn, bn, out4, idx4)
        s3, b3 = self.decoder3(s4, b4, out3, idx3)
        s2, b2 = self.decoder1(s3, b3, out2, idx2)
        s1, b1 = self.decoder1(s2, b2, out1, idx1)

        fine = torch.cat((out1, self.up(s2)), dim=1)
        coarse = self.classifier1(s1)
        dst_map = self.classifier2(b1)

        # fine:细粒度特征图 coarse:粗预测概率图 dst_map:距离图
        return fine, coarse, dst_map

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

    def get_grad_param(self):
        return self.parameters()


# 残差U-Net的结构
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), 1, (1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3, 3), 1, (1, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, input):
        out = self.prelu(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        out = self.prelu(self.bn2(out + input))
        return out


class EncoderBlock(BasicBlock):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__(in_ch=in_ch, out_ch=out_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        # out 这里是跳跃连接的特征
        out = super(EncoderBlock, self).forward(input)
        # out_encoder是下采样后的特征，indices返回最大索引所在的位置
        out_encoder, indices = self.maxpool(out)
        return out_encoder, out, indices


class FIFDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FIFDecoderBlock, self).__init__()
        self.uppool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Conv2d(in_ch, out_ch, (3, 3), 1, (1, 1), bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.update1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), 1, (1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, (3, 3), 1, (1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.update2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), 1, (1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, (3, 3), 1, (1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.transform1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.transform2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    # input1为layer路径的输入，input2为boundary路径的输出，跳跃连接的输出，indices是索引
    def forward(self, input1, input2, out_block, indices):
        # 上采样后的特征
        uppool1 = self.uppool(input1, indices)
        uppool2 = self.uppool(input2, indices)
        # layer路径中上采样和跳跃连接concat之后的特征
        concatS = torch.cat((out_block, uppool1), dim=1)
        updateS = self.update1(concatS)
        concatB = torch.cat((updateS, uppool2), dim=1)
        updateB = self.update2(concatB)
        outS = self.prelu(self.transform1(concatS) + updateS + self.bn(self.conv(torch.cat((updateS, updateB), dim=1))))
        outB = self.prelu(updateB + self.transform2(concatB))
        return outS, outB


class ClassifierBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, input):
        out_conv = self.conv(input)
        return out_conv
