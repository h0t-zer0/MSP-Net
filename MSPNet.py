import torch
import torch.nn as nn
import torch.nn.functional as F
from models.smt import smt_t


# Local Perception Unit
class LPU(nn.Module):
    def __init__(self, channels, kernel=3):
        super(LPU, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel), stride=1, padding=(0, kernel//2), bias=False),
            nn.Conv2d(channels, channels, kernel_size=(kernel, 1), stride=1, padding=(kernel//2, 0), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=kernel//2, padding=kernel//2, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=1, padding=kernel//2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = self.final_conv(torch.cat([x1, x2], dim=1))
        return out


# Dynamic Perception Module
class DPM(nn.Module):
    def __init__(self, channels):
        super(DPM, self).__init__()
        self.g2l = nn.Sequential(
            LPU(channels, kernel=5),
            LPU(channels, kernel=3),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.l2g = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            LPU(channels, kernel=3),
            LPU(channels, kernel=5)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.g2l(x)
        x2 = self.l2g(x)
        out = self.final_conv(torch.cat([x1, x2], dim=1))
        return out  


# Mixed Perception Layer
class MPLayer(nn.Module):
    def __init__(self, channels):
        super(MPLayer, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.branch_2 = LPU(channels, kernel=3)
        self.branch_3 = LPU(channels, kernel=5)
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.final_gelu = nn.GELU()

    def forward(self, x):
        res = x
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        out = self.final_conv(torch.cat([x1, x2, x3], dim=1))
        out = out + res
        return self.final_gelu(out)
    

# Mixed Perception Module
class MPM(nn.Module):
    def __init__(self, channels):
        super(MPM, self).__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            MPLayer(channels)
        )
        self.branch_1 = nn.Sequential(
            LPU(channels, kernel=3),
            MPLayer(channels)
        )
        self.branch_2 = nn.Sequential(
            LPU(channels, kernel=5),
            MPLayer(channels)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.final_gelu = nn.GELU()

    def forward(self, x):
        res = x
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = self.final_conv(torch.cat([x0, x1, x2], dim=1))
        out = out + res
        return self.final_gelu(out)
    

# Fusion Module
class fusion(nn.Module):
    def __init__(self, channels):
        super(fusion, self).__init__()
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.max_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(dim=1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.final_gelu = nn.GELU()
    
    def forward(self, x):
        res = x
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        avg_out = self.avg_pooling(x1)  # [B, C, 1, 1]
        max_out = self.max_pooling(x2)  # [B, C, 1, 1]
        avg_score = self.avg_conv(torch.cat([avg_out, max_out], dim=1))  # [B, C, 1, 1]
        max_score = self.max_conv(torch.cat([avg_out, max_out], dim=1))  # [B, C, 1, 1]
        out = torch.cat([x1 * avg_score, x2 * max_score], dim=1)
        out = self.final_conv(out)
        out = out + res
        return self.final_gelu(out)


class MSPNet(nn.Module):
    def __init__(self):
        super(MSPNet, self).__init__()

        self.smt = smt_t()

        self.Translayer1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )
        self.Translayer2_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )
        self.Translayer3_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )
        self.Translayer4_1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )

        self.fusion1 = fusion(channels=64)
        self.fusion2 = fusion(channels=64)
        self.fusion3 = fusion(channels=64)

        self.DPM1 = DPM(channels=64)
        self.DPM2 = DPM(channels=64)
        self.DPM3 = DPM(channels=64)
        self.DPM4 = DPM(channels=64)
        self.MPM1 = MPM(channels=64)
        self.MPM2 = MPM(channels=64)
        self.MPM3 = MPM(channels=64)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feature_list = self.smt(x)

        r4 = feature_list[3]  # C = 512
        r3 = feature_list[2]  # C = 256
        r2 = feature_list[1]  # C = 128
        r1 = feature_list[0]  # C = 64

        r1 = self.Translayer1_1(r1)
        r2 = self.Translayer2_1(r2)
        r3 = self.Translayer3_1(r3)
        r4 = self.Translayer4_1(r4)
        
        # local
        r4 = self.DPM4(r4)
        r3 = self.DPM3(r3 + self.up1(r4))
        r2 = self.DPM2(r2 + self.up1(r3))
        r1 = self.DPM1(r1 + self.up1(r2))

        r4 = self.up1(r4)
        r34 = self.fusion3(r3 + r4)
        r3 = self.up1(r3)
        r23 = self.fusion2(r2 + r3)
        r2 = self.up1(r2)
        r12 = self.fusion1(r1 + r2)

        s3 = self.MPM3(r4 + r34)
        s3 = self.up1(s3)
        s2 = self.MPM2(s3 + r23)
        s2 = self.up1(s2)
        s1 = self.MPM1(s2 + r12)

        s_final = F.interpolate(self.predtrans1(s1), size=416, mode='bilinear')
        s_p1 = F.interpolate(self.predtrans2(s2), size=416, mode='bilinear')
        s_p2 = F.interpolate(self.predtrans3(s3), size=416, mode='bilinear')

        return s_final, s_p1, s_p2

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    # flops, params = profile(MSPNet(x), (x,))
    # print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
