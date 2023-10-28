import torch
from torch import nn
import torch.nn.functional as F
import common


def make_model(args):
    return non_local_MEM(args.n_colors, args.n_feats)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.ksize = kernel_size
        self.bias = bias
        self.bn = bn
        self.n_feats = n_feats

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

    def flops(self, input_shape):
        b, c, h, w = input_shape
        flops = 0
        flops += self.ksize ** 2 * self.n_feats * h * w * self.n_feats * 2
        if self.bias:
            flops += self.n_feats * h * w * 2
        if self.bn:
            flops += 2 * b * c * h * w * 2
        return flops


class MultiResBlocks(nn.Module):
    def __init__(self, n_feats, num_blocks=4):
        super(MultiResBlocks, self).__init__()
        self.num_blocks = num_blocks
        blocks = [ResBlock(common.default_conv, n_feats=n_feats, kernel_size=3, act=nn.PReLU(), res_scale=1)
                  for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

    def flops(self, input_shape):
        return self.num_blocks * self.blocks[0].flops(input_shape)


class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 is_training=True,
                 sub_sample=True):
        super(_NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = 64
        self.inter_channels = self.in_channels // 2
        self.is_training = is_training

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # mem
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256))
        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        b, c, h, w = x.size()  # [1, 64,32,32]

        g_x = self.g(x).view(b, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.inter_channels, -1)  # (1, 32, 1024)
        theta_x = theta_x.permute(0, 2, 1)  # (1, 1024, 32)

        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        phi_x_for_quant = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)  # (1, 64, 32, 32)

        # mem
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # (1, 32, 256)
        f1 = torch.matmul(phi_x_for_quant, mbg)  # (1, 256, 256)
        f_div_C1 = F.softmax(f1 * (int(self.inter_channels) ** (-0.5)), dim=-1)
        y1 = torch.matmul(f_div_C1, mbg.permute(0, 2, 1))  # yt = softmax(tm*m^) # (1, 256, 32)
        y1 = y1.permute(0, 2, 1).view(b, self.inter_channels, h, w).contiguous()
        W_y1 = self.W_z1(y1)

        z = W_y + x + W_y1

        return z


class non_local_MEM(nn.Module):
    def __init__(self, in_dim, num_feats):
        super(non_local_MEM, self).__init__()
        self.num_feats = num_feats
        self.in_dim = in_dim
        self.conv_head = nn.Conv2d(in_dim, num_feats, 3, 1, 1)
        self.op1 = nn.Sequential(MultiResBlocks(64, 4),
                                 _NonLocalBlockND(is_training=True)
                                 )
        self.op2 = nn.Sequential(MultiResBlocks(64, 4),
                                 _NonLocalBlockND(is_training=True)
                                 )
        self.op3 = nn.Sequential(MultiResBlocks(64, 4),
                                 _NonLocalBlockND(is_training=True)
                                 )
        self.conv_tail = nn.Sequential(MultiResBlocks(64, 4),
                                       nn.Conv2d(num_feats, in_dim, 3, 1, 1))
        self.downsamples = nn.ModuleList([Downsample(num_feats, True) for _ in range(2)])
        self.upsample1 = Upsample(num_feats, True)
        self.upsample2 = Upsample(num_feats, True)
        self.conv = nn.Conv2d(in_dim, num_feats, 2, 1, 1)
        print("Construct non_local_MEM ...")

    def forward(self, x):
        pyramid_level = [self.conv_head(x), ]
        for i in range(2):
            # pyramid_level.insert(0, F.avg_pool2d(pyramid_level[0], 2, 2))
            pyramid_level.insert(0, self.downsamples[i](pyramid_level[0]))
        for i in range(3):
            if i == 0:
                fea = self.op1(pyramid_level[i])
            elif i == 1:
                # fea = self.op2(pyramid_level[i]) + F.interpolate(input=fea, scale_factor=2, mode='bilinear', align_corners=True)
                fea = self.op2(pyramid_level[i]) + self.upsample1(fea)  # 72
            elif i == 2:
                fea = self.op3(pyramid_level[i]) + self.upsample2(fea)
        res = self.conv_tail(fea)
        return x + res


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# if __name__ == '__main__':
#     from thop import profile
#     module = MANA(30, 64)
#     num_params = 0
#     for param in module.parameters():
#         num_params += param.numel()
#     print('Total number of parameters: %.2f Mb' % (num_params / 1e6))
#     # swin(x)

if __name__ == '__main__':
    a = torch.randn(1, 30, 128, 128).cuda()
    model = non_local_MEM(30, 64).cuda()
    b = model(a)
    print(b.shape)
