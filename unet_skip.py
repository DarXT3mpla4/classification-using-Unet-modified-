import torch
from torchsummary import summary
import torch.nn as nn
from contextlib import redirect_stdout


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        # nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_c),
        # nn.ReLU(inplace=True),
    )
    return conv


def residual_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
    )
    return conv


def identity_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, ),
        nn.BatchNorm2d(out_c),
    )
    return conv


# def SE_block(in_c, out_c, r):


def up_conv_bilinear(in_c, out_c):
    conv = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_c, out_c, kernel_size=1),
        # nn.BatchNorm2d(out_c),
        # nn.ReLU(inplace=True),)
    )
    return conv


def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(UNet, self).__init__()
        # self.final_x = torch.empty(256,256)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_inter = nn.ReLU(inplace=True)

        # down sampling
        self.down_conv_1 = double_conv(n_channels, 16)
        self.down_conv_2 = double_conv(16, 32)
        self.down_conv_3 = double_conv(32, 64)
        self.down_conv_4 = double_conv(64, 128)
        self.down_conv_5 = double_conv(128, 256)

        # up transpose
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_1 = up_conv_bilinear(256,128)
        self.up_conv_1 = double_conv(256, 128)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_2 = up_conv_bilinear(128,64)
        self.up_conv_2 = double_conv(128, 64)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_3 = up_conv_bilinear(64,32)
        self.up_conv_3 = double_conv(64, 32)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=32,
                                             out_channels=16,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_4 = up_conv_bilinear(32, 16)
        self.up_conv_4 = double_conv(32, 16)

        self.inter_identity_stage1 = identity_block(16, 16)
        self.inter_residual_input = residual_block(16, 16)

        # 2nd Iteration

        # self.down_conv_1 = double_conv(3,32)
        self.down_conv_6 = double_conv(16, 32)
        self.down_conv_7 = double_conv(32, 64)
        self.down_conv_8 = double_conv(64, 128)
        self.down_conv_9 = double_conv(128, 256)

        # up transpose
        self.up_trans_5 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_5 = up_conv_bilinear(256, 128)
        self.up_conv_5 = double_conv(256, 128)

        self.up_trans_6 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_6 = up_conv_bilinear(128, 64)
        self.up_conv_6 = double_conv(128, 64)

        self.up_trans_7 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_7 =  up_conv_bilinear(64, 32)
        self.up_conv_7 = double_conv(64, 32)

        self.up_trans_8 = nn.ConvTranspose2d(in_channels=32,
                                             out_channels=16,
                                             kernel_size=2,
                                             stride=2)
        # self.up_trans_8 = up_conv_bilinear(32, 16)
        self.up_conv_8 = double_conv(32, 16)

        self.out = nn.Conv2d(
            in_channels=16,
            out_channels=self.n_classes,
            kernel_size=1)

    def forward(self, image):
        # encoder part
        # bs, c, h, w

        x1 = self.down_conv_1(image)  #
        x2 = self.max_pool_2x2(x1)  # 16 channels

        x3 = self.down_conv_2(x2)  #
        x4 = self.max_pool_2x2(x3)  # 32 channels

        x5 = self.down_conv_3(x4)  #
        x6 = self.max_pool_2x2(x5)  # 64 channels

        x7 = self.down_conv_4(x6)  #

        x8 = self.max_pool_2x2(x7)  # 128 channels

        x9 = self.down_conv_5(x8)  #

        # decoder part
        x = self.up_trans_1(x9)

        # y = crop_tensor(x7, x)
        x_up1 = self.up_conv_1(torch.cat([x, x7], 1))  # before max-pooling and result of deconv

        # nn.Upsample(mode='bilinear', scale_factor=2)

        x = self.up_trans_2(x_up1)

        # print(f'The size after up transpose 2 {x.size()}')
        # print(f'The size of x5 is {x5.size()}')
        # y = crop_tensor(x5, x)
        x_up2 = self.up_conv_2(torch.cat([x, x5], 1))
        # nn.Upsample(mode='bilinear', scale_factor=2)
        # print(f'The size after concatenation 1 {x.size()}')

        x = self.up_trans_3(x_up2)
        # y = crop_tensor(x3, x)

        x_up3 = self.up_conv_3(torch.cat([x, x3], 1))
        # print(f'Tensor size at THIRD concatenation {x.size()}')

        x_last = self.up_trans_4(x_up3)  # first stage output
        # y = crop_tensor(x1, x)
        # print(f'Tensor size at x_last {x_last.size()}')

        x_stage1 = self.up_conv_4(torch.cat([x_last, x1], 1))
        # print(f'Tensor size at stage 1 {x_stage1.size()}')

        # intermediate_x =self.up_conv_5(torch.cat([x, x1], 1))
        # print(f'The size of x1 {x1.size()}')
        intermediate_x = x_stage1
        test_var = self.inter_identity_stage1(intermediate_x) + self.inter_residual_input(x1)
        print(f'size of test_var is {test_var.size()}')
        print(f'size of x1 {x1.size()}')
        # print(f'The size of test variable {test_var.size()}')
        # intermediate_x = torch.cat([self.inter_identity_stage1(intermediate_x), self.inter_identity_input(x1)],1)
        # print(f'The size of intermediate_x {intermediate_x.size()}')
        intermediate_x = test_var
        # print(f'The size before 2nd iteration {intermediate_x.size()}')

        # print(f'intermediate_x size {intermediate_x.size()}')
        x2 = self.relu_inter(intermediate_x)
        x2 = self.max_pool_2x2(x2)  #
        # print(f'size of x2 is {x2.size()}')

        # print(f'size of x_up3 {x_up3.size()}')#
        # x3 = torch.cat([x2, x_up3], 1)
        x3 = self.down_conv_6(x2)
        x4 = self.max_pool_2x2(x3)

        #
        # x5 = torch.cat([x5, x_up2], 1)
        x5 = self.down_conv_7(x4)
        x6 = self.max_pool_2x2(x5)

        # x7 = torch.cat([x7, x_up1], 1)
        x7 = self.down_conv_8(x6)  #
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv_9(x8)  #

        # decoder part
        x = self.up_trans_5(x9)
        # y = crop_tensor(x7, x)
        x = self.up_conv_5(torch.cat([x, x7], 1))

        x = self.up_trans_6(x)
        # y = crop_tensor(x5, x)
        x = self.up_conv_6(torch.cat([x, x5], 1))

        x = self.up_trans_7(x)
        # y = crop_tensor(x3, x)
        x = self.up_conv_7(torch.cat([x, x3], 1))

        x = self.up_trans_8(x)
        # y = crop_tensor(x1, x)
        x = self.up_conv_8(torch.cat([x, x1], 1))

        y = self.out(x)
        return y


if __name__ == "__main__":
    # image = torch.rand((1,3,256,256))
    net = UNet(2, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    # plot_net(net, to_file="image.png")
    # summary(net, (3,544,384), 1)
    print(summary(net, (3, 544, 384), 1))
    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            summary(net, (3, 544, 384), 1)
