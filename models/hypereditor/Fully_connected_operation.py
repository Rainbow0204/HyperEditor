import torch
from torch.nn import Module, Linear


class SeparableBlock(Module):

    def __init__(self, input_size, kernel_channels_in, kernel_channels_out, kernel_size):
        super(SeparableBlock, self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.kernel_channels_in = kernel_channels_in
        self.kernel_channels_out = kernel_channels_out

        self.make_kernel_in = Linear(input_size, kernel_size * kernel_size * kernel_channels_in)
        self.make_kernel_out = Linear(input_size, kernel_size * kernel_size * kernel_channels_out)

        self.kernel_linear_in = Linear(kernel_channels_in, kernel_channels_in)
        self.kernel_linear_out = Linear(kernel_channels_out, kernel_channels_out)

    def forward(self, features):

        features = features.view(-1, self.input_size)

        kernel_in = self.make_kernel_in(features).view(-1, self.kernel_size, self.kernel_size, 1, self.kernel_channels_in)
        kernel_out = self.make_kernel_out(features).view(-1, self.kernel_size, self.kernel_size, self.kernel_channels_out, 1)

        kernel = torch.matmul(kernel_out, kernel_in)

        kernel = self.kernel_linear_in(kernel).permute(0, 1, 2, 4, 3)
        kernel = self.kernel_linear_out(kernel)
        kernel = kernel.permute(0, 4, 3, 1, 2)

        return kernel
