import torch
import math
from torch.cuda.amp import autocast
from utils.UnarySimUtils import *
import torch
import math

from torch.cuda.amp import autocast

import torch


class Conv2SCONNA(torch.nn.Conv2d):
    """
    This module is for convolution with unary input and output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        mode="unipolar",
        scaled=True,
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(Conv2SCONNA, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        self.mode = mode
        self.scaled = scaled
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

    def FSUKernel_accumulation(self, input):
        output_size = conv2d_output_shape(
            (input.size()[2], input.size()[3]),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            pad=self.padding,
            stride=self.stride,
        )

        if True in self.even_cycle_flag:
            input_padding = self.padding_0(input)
        else:
            input_padding = self.padding_1(input)

        # if unipolar mode, even_cycle_flag is always False to pad 0.
        self.even_cycle_flag.data = self.bipolar_mode ^ self.even_cycle_flag

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(
            input_padding, self.kernel_size, self.dilation, 0, self.stride
        )
        print("Un Fold Image :", input_im2col.shape)
        print("Un fold Image :", input_im2col)
        input_transpose = input_im2col.transpose(1, 2)

        input_reshape = input_transpose.reshape(-1, 1, input_transpose.size()[-1])

        # first dim should always be batch
        batch = input_reshape.size()[0]

        # generate weight and bias bits for current cycle
        weight_bs = self.weight_bsg(self.weight_rng_idx).type(torch.float)
        if weight_bs.size()[0] != batch:
            weight_bs = torch.cat(batch * [weight_bs], 0)
            self.weight_rng_idx = torch.cat(batch * [self.weight_rng_idx], 0)
        torch.add(
            self.weight_rng_idx, input_reshape.type(torch.long), out=self.weight_rng_idx
        )

        kernel_out = torch.empty(0, device=input.device)
        torch.matmul(
            input_reshape.type(torch.float), weight_bs.transpose(1, 2), out=kernel_out
        )
        kernel_out.squeeze_(1)

        kernel_out_reshape = kernel_out.reshape(
            input.size()[0], -1, kernel_out.size()[-1]
        )
        kernel_out_transpose = kernel_out_reshape.transpose(1, 2)
        kernel_out_fold = torch.nn.functional.fold(
            kernel_out_transpose, output_size, (1, 1)
        )

        if self.has_bias is True:
            bias_bs = self.bias_bsg(self.bias_rng_idx).type(torch.float)
            self.bias_rng_idx.add_(1)
            kernel_out_fold += bias_bs.view(1, -1, 1, 1).expand_as(kernel_out_fold)

        if self.mode == "unipolar":
            return kernel_out_fold

        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            weight_bs_inv = 1 - self.weight_bsg_inv(self.weight_rng_idx_inv).type(
                torch.float
            )
            if weight_bs_inv.size()[0] != batch:
                weight_bs_inv = torch.cat(batch * [weight_bs_inv], 0)
                self.weight_rng_idx_inv = torch.cat(
                    batch * [self.weight_rng_idx_inv], 0
                )
            torch.add(
                self.weight_rng_idx_inv,
                1 - input_reshape.type(torch.long),
                out=self.weight_rng_idx_inv,
            )

            kernel_out_inv = torch.empty(0, device=input.device)
            torch.matmul(
                1 - input_reshape.type(torch.float),
                weight_bs_inv.transpose(1, 2),
                out=kernel_out_inv,
            )
            kernel_out_inv.squeeze_(1)

            kernel_out_reshape_inv = kernel_out_inv.reshape(
                input.size()[0], -1, kernel_out_inv.size()[-1]
            )
            kernel_out_transpose_inv = kernel_out_reshape_inv.transpose(1, 2)
            kernel_out_fold_inv = torch.nn.functional.fold(
                kernel_out_transpose_inv, output_size, (1, 1)
            )

            return kernel_out_fold + kernel_out_fold_inv

    @autocast()
    def forward(self, input):
        kernel_out_total = self.FSUKernel_accumulation(input)
        self.accumulator.data = self.accumulator.add(kernel_out_total)
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(self.stype)
