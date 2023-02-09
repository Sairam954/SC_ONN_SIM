import torch
import math
from torch.cuda.amp import autocast
from utils.UnarySimUtils import *
import torch
import math

from torch.cuda.amp import autocast

import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class FSUAdd(torch.nn.Module):
    """
    This module is for unary addition for arbitrary scale, including scaled/non-scaled, unipolar/bipolar.
    """

    def __init__(
        self,
        mode="bipolar",
        scaled=True,
        scale=None,
        dim=0,
        depth=10,
        entry=None,
        stype=torch.float,
    ):
        super(FSUAdd, self).__init__()

        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # scale is an arbitrary value that larger than 0
        self.scale = scale
        # dimension to do reduced sum
        self.dim = dim
        # max value in the accumulator
        self.acc_max = 2 ** (depth - 2)
        # min value in the accumulator
        self.acc_min = -(2 ** (depth - 2))
        # number of entries in dim to do reduced sum
        self.entry = entry
        self.stype = stype

        # the carry scale at the output
        self.scale_carry = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulator for (PC - offset)
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.first = True

    def forward(self, input, scale=None, entry=None):
        if self.first:
            if self.scaled is True:
                if self.scale is None:
                    self.scale_carry.fill_(input.size()[self.dim])
                else:
                    self.scale_carry.fill_(self.scale)
                if scale is not None:
                    # runtime scale will override the default value
                    self.scale_carry.fill_(scale)
            if self.mode == "bipolar":
                if self.entry is None:
                    self.offset.data = (input.size()[self.dim] - self.scale_carry) / 2
                else:
                    self.offset.data = (self.entry - self.scale_carry) / 2
                if entry is not None:
                    # runtime entry will override the default value in bipolar mode
                    self.offset.data = (entry - self.scale_carry) / 2
            self.first = False
        acc_delta = torch.sum(input.type(torch.float), self.dim) - self.offset
        self.accumulator.data = self.accumulator.add(acc_delta).clamp(
            self.acc_min, self.acc_max
        )
        output = torch.gt(self.accumulator, self.scale_carry).type(torch.float)
        self.accumulator.sub_(output * self.scale_carry).clamp_(
            self.acc_min, self.acc_max
        )
        return output.type(self.stype)


class FSUAdduGEMM(torch.nn.Module):
    """
    This module is for unary addition in uGEMM, including scaled/non-scaled, unipolar/bipolar.
    """

    def __init__(self, mode="bipolar", scaled=True, dim=0, stype=torch.float):
        super(FSUAdduGEMM, self).__init__()

        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # dimension to do reduce sum
        self.dim = torch.nn.Parameter(
            torch.zeros(1).type(torch.int8), requires_grad=False
        )
        self.dim.fill_(dim)
        self.stype = stype

        # upper bound for accumulation counter in scaled mode
        # it is the number of inputs, e.g., the size along the dim dimension
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

    def forward(self, input):
        self.acc_bound.fill_(input.size()[self.dim.item()])
        if self.mode == "bipolar":
            self.offset.fill_((self.acc_bound.item() - 1) / 2)
        self.accumulator.data = self.accumulator.add(
            torch.sum(input.type(torch.float), self.dim.item())
        )

        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(self.stype)


class GainesAdd(torch.nn.Module):
    """
    this module is for Gaines addition.
    1) MUX for scaled addition
    2) OR gate for non-scaled addition
    """

    def __init__(
        self,
        mode="bipolar",
        scaled=True,
        dim=0,
        rng="Sobol",
        rng_dim=5,
        rng_width=8,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(GainesAdd, self).__init__()

        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        if self.mode == "bipolar" and self.scaled is False:
            raise ValueError(
                "Non-scaled addition for biploar data is not supported in Gaines approach."
            )
        # dimension to do reduce sum
        self.dim = dim
        self.stype = stype
        self.rng = RNG(rng_width, rng_dim, rng, rtype=rtype)()
        self.rng_idx = torch.nn.Parameter(
            torch.zeros(1).type(torch.long), requires_grad=False
        )

    def forward(self, input):
        if self.scaled is True:
            randNum = self.rng[self.rng_idx.item()]
            assert (
                randNum.item() < input.size()[self.dim]
            ), "randNum should be smaller than the dimension size of addition."
            # using a MUX for both unipolar and bipolar
            output = torch.unbind(
                torch.index_select(input, self.dim, randNum.type(torch.long).view(1)),
                self.dim,
            )[0]
            self.rng_idx.data = self.rng_idx.add(1) % self.rng.numel()
        else:
            # only support unipolar data using an OR gate
            output = torch.gt(torch.sum(input, self.dim), 0)

        return output.type(self.stype)


class FSUConv2d(torch.nn.Module):
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
        mode="bipolar",
        scaled=True,
        scale=None,
        depth=12,
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(FSUConv2d, self).__init__()

        self.stype = stype
        self.PC = FSUConv2dPC(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            binary_weight=binary_weight,
            binary_bias=binary_bias,
            bitwidth=bitwidth,
            mode=mode,
            btype=btype,
            rtype=rtype,
            stype=stype,
        )
        if scaled is True:
            if scale is None:
                scale_add = math.prod(num2tuple(kernel_size)) * in_channels + bias
            else:
                scale_add = scale
        else:
            scale_add = 1.0
        self.ACC = FSUAdd(
            mode=mode,
            scaled=scaled,
            scale=scale_add,
            dim=0,
            depth=depth,
            entry=math.prod(num2tuple(kernel_size)) * in_channels + bias,
            stype=stype,
        )

    @autocast()
    def forward(self, input, scale=None, entry=None):
        pc = self.PC(input)
        output = self.ACC(pc.unsqueeze(0), scale, entry)
        return output.type(self.stype)


class FSUConv2dPC(torch.nn.Conv2d):
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
        mode="bipolar",
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(FSUConv2dPC, self).__init__(
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
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == "zeros", "Supported padding_mode number is 'zeros'."

        self.mode = mode

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = SourceGen(
                binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        if bias and (binary_bias is not None):
            self.bias.data = SourceGen(
                binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        # define the kernel linear
        self.weight_bsg = BSGen(
            self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype
        )
        self.weight_rng_idx = torch.nn.Parameter(
            torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
        ).view(1, self.weight.size()[0], -1)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(
                torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(
                self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype
            )
            self.weight_rng_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
            ).view(1, self.weight.size()[0], -1)

        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(
            torch.ones(1, dtype=torch.bool), requires_grad=False
        )
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(
            torch.tensor([self.mode == "bipolar"], dtype=torch.bool),
            requires_grad=False,
        )

    def FSUConv2d_PC(self, input):
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
        return self.FSUConv2d_PC(input).type(self.stype)


class FSUConv2duGEMM(torch.nn.Conv2d):
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
        super(FSUConv2duGEMM, self).__init__(
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

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == "zeros", "Supported padding_mode number is 'zeros'."

        # upper bound for accumulation counter in scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(math.prod(num2tuple(self.kernel_size)) * in_channels)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_(
                (math.prod(num2tuple(self.kernel_size)) * in_channels - 1) / 2
            )
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("FSUConv2d mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = SourceGen(
                binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        if bias and (binary_bias is not None):
            self.bias.data = SourceGen(
                binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        # define the kernel linear
        self.weight_bsg = BSGen(
            self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype
        )
        self.weight_rng_idx = torch.nn.Parameter(
            torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
        ).view(1, self.weight.size()[0], -1)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(
                torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(
                self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype
            )
            self.weight_rng_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
            ).view(1, self.weight.size()[0], -1)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(
            torch.ones(1, dtype=torch.bool), requires_grad=False
        )
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(
            torch.tensor([self.mode == "bipolar"], dtype=torch.bool),
            requires_grad=False,
        )

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


class Conv2dSCONNA(torch.nn.Conv2d):
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
        # * calling  the super class method and intiailizing properties
        super(Conv2dSCONNA, self).__init__(
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

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == "zeros", "Supported padding_mode number is 'zeros'."

        # upper bound for accumulation counter in scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(math.prod(num2tuple(self.kernel_size)) * in_channels)

        if bias is True:
            self.acc_bound.add_(1)
        print("Upper Bound for accumulation ", self.acc_bound)
        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_(
                (math.prod(num2tuple(self.kernel_size)) * in_channels - 1) / 2
            )
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("FSUConv2d mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = SourceGen(
                binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()
        print("Weight Dimension :", self.weight.data.shape)
        if bias and (binary_bias is not None):
            self.bias.data = SourceGen(
                binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()
        print("Bias Dimension :", self.bias.data.shape)
        # define the kernel linear
        self.weight_bsg = BSGen(
            self.weight.view(1, self.weight.size()[0], -1),
            self.rng,
            stype=stype,
        )
        self.weight_rng_idx = torch.nn.Parameter(
            torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
        ).view(1, self.weight.size()[0], -1)
        print("Random Weigth IDX  shape :", self.weight_rng_idx.shape)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(
                torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(
                self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype
            )
            self.weight_rng_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
            ).view(1, self.weight.size()[0], -1)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(
            torch.ones(1, dtype=torch.bool), requires_grad=False
        )
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(
            torch.tensor([self.mode == "bipolar"], dtype=torch.bool),
            requires_grad=False,
        )
        # self.weight_bit_stream = torch.empty(2**self.bitwidth)
        # self.bias_bit_stream = torch.empty(2**self.bitwidth)
        # print("Weight Bit Stream Shape :", self.weight_bit_stream.shape)

        # for i in range(2**self.bitwidth):
        #     idx = torch.zeros(self.weight.data.size()).type(torch.long).to(device)
        #     self.weight_bit_stream[i] = self.weight_bsg(idx + i)
        # # print("Weight BS :", self.weight_bit_stream.shape)
        # for i in range(2**self.bitwidth):
        #     idx = torch.zeros(self.bias.data.size()).type(torch.long).to(device)
        #     self.weight_bit_stream[i] = self.bias_bsg(idx + i)
        # print("Bias BS :", self.weight_bit_stream)
        # print("Weight BS :", self.weight_bit_stream.shape)

    def FSUKernel_accumulation(self, input):
        output_size = conv2d_output_shape(
            (input.size()[2], input.size()[3]),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            pad=self.padding,
            stride=self.stride,
        )
        print("Output Size :", output_size)
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
        print("Input IM 2 Col :", input_im2col.shape)
        input_transpose = input_im2col.transpose(1, 2)
        print("Input Transpose :", input_transpose.shape)
        input_reshape = input_transpose.reshape(-1, 1, input_transpose.size()[-1])
        print("input_reshape :", input_reshape.shape)
        # first dim should always be batch
        batch = input_reshape.size()[0]
        print("Batch :", batch)
        # generate weight and bias bits for current cycle
        # print("Random Number weight Id :", self.weight_rng_idx)
        # * array for declaring weight corresponding to each bit stream
        self.weight_bit_stream = torch.empty(
            2**self.bitwidth,
            batch,
            self.weight.data.shape[0],
            self.weight.data.shape[1]
            * self.weight.data.shape[2]
            * self.weight.data.shape[3],
        )
        print("Weight Bit Stream :", self.weight_bit_stream.shape)
        self.bias_bit_stream = torch.empty(2**self.bitwidth, self.bias.data.shape[0])
        print("Bias Bit Stream :", self.bias_bit_stream.shape)
        for idx in range(2**self.bitwidth):
            weight_bs = self.weight_bsg(self.weight_rng_idx + idx).type(torch.float)
            print("Weight Binary Stochastic :", weight_bs.shape)

            if weight_bs.size()[0] != batch:
                weight_bs = torch.cat(batch * [weight_bs], 0)
                self.weight_rng_idx = torch.cat(batch * [self.weight_rng_idx], 0)
            self.weight_bit_stream[idx] = weight_bs
        print("Weight Bitstream Size :", self.weight_bit_stream.shape)
        torch.add(
            self.weight_rng_idx, input_reshape.type(torch.long), out=self.weight_rng_idx
        )
        print("Weight Binary Stochastic :", weight_bs[0])
        print("Binary to Stochastic Weight ", self.weight_rng_idx)

        kernel_out = torch.empty(0, device=input.device)
        input_reshape = input_reshape.repeat(
            2**self.bitwidth,
            1,
            1,
            1,
        )
        print("Input reshape :", input_reshape.shape)
        print("Weight Transpose :", self.weight_bit_stream.transpose(2, 3).shape)
        print("Weight Bitstream Size :", self.weight_bit_stream.shape)
        torch.matmul(
            input_reshape.type(torch.float),
            self.weight_bit_stream.transpose(2, 3),
            out=kernel_out,
        )
        # torch.matmul(
        #     input_reshape.type(torch.float),
        #     weight_bs.transpose(1, 2),
        #     out=kernel_out,
        # )
        print("Kernel Out Shape :", kernel_out[0])
        kernel_out.squeeze_(2)
        print("Kernel Out Shape :", kernel_out.shape)
        kernel_out_reshape = kernel_out.reshape(
            2**self.bitwidth, input.size()[0], -1, kernel_out.size()[-1]
        )
        # kernel_out_reshape = kernel_out.reshape(
        #     input.size()[0], -1, kernel_out.size()[-1]
        # )
        print("Kernel Out Reshape :", kernel_out_reshape.shape)
        kernel_out_transpose = kernel_out_reshape.transpose(2, 3)
        # kernel_out_transpose = kernel_out_reshape.transpose(1, 2)
        print("Kernel Out Transpose :", kernel_out_transpose.shape)
        kernel_out_fold = torch.empty(
            2**self.bitwidth,
            1,
            kernel_out_transpose.shape[2],
            output_size[0],
            output_size[1],
        )
        print()
        for idx in range(2**self.bitwidth):
            kernel_out_fold[idx] = torch.nn.functional.fold(
                kernel_out_transpose[idx], output_size, (1, 1)
            )
            print("Kernel Out Fold  :", kernel_out_fold.shape)
            if self.has_bias is True:
                bias_bs = self.bias_bsg(self.bias_rng_idx + idx).type(torch.float)
                # self.bias_rng_idx.add_(1)
                kernel_out_fold[idx] += bias_bs.view(1, -1, 1, 1).expand_as(
                    kernel_out_fold[idx]
                )
        print("Kernel_out_fold", kernel_out_fold[0])
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


class HUBConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
    This cycle is the mac cycle using unipolar umul, i.e., half the bipolar umul.
    As such, cycle = 2 ^ (bitwidth - 1).
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
        rng="Sobol",
        cycle=128,
        rounding="round",
    ):
        super(HUBConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == "zeros", "Supported padding_mode number is 'zeros'."

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight

        if bias and (binary_bias is not None):
            self.bias.data = binary_bias

        # mac computing cycle
        self.cycle = cycle

        # bitwidth of rng
        self.bitwidth = (self.cycle - 1).bit_length()

        # random_sequence from sobol RNG
        self.irng = RNG(self.bitwidth, 1, rng)()
        self.wrng = RNG(self.bitwidth, 1, "Sobol")()

        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the rng
        self.input_map = torch.nn.Parameter(torch.empty(cycle), requires_grad=False)
        input_val_cycle = torch.empty(0)
        torch.cat(
            cycle
            * [
                torch.as_tensor([c for c in range(cycle)], dtype=torch.float).unsqueeze(
                    1
                )
            ],
            1,
            out=input_val_cycle,
        )
        input_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.irng.unsqueeze(0), out=input_bit_cycle)
        self.input_map.data = torch.sum(input_bit_cycle, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the rng, under a specific input and weight
        self.wght_map = torch.nn.Parameter(
            torch.empty(cycle, cycle), requires_grad=False
        )
        wght_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.wrng.unsqueeze(0), out=wght_bit_cycle)
        for c in range(cycle):
            self.wght_map.data[c] = torch.sum(
                wght_bit_cycle[:, 0 : self.input_map.data[c]], 1
            ).squeeze_()

        # rounding mode
        self.rounding = rounding

        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None

    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            input_max_int = input.abs().max().log2()
            wght_max_int = self.weight.abs().max().log2()
            if self.rounding == "round":
                input_max_int = input_max_int.round()
                wght_max_int = wght_max_int.round()
            elif self.rounding == "floor":
                input_max_int = input_max_int.floor()
                wght_max_int = wght_max_int.floor()
            elif self.rounding == "ceil":
                input_max_int = input_max_int.ceil()
                wght_max_int = wght_max_int.ceil()

            self.rshift_input = input_max_int - self.bitwidth
            self.rshift_wght = wght_max_int - self.bitwidth
            self.rshift_output = self.bitwidth - input_max_int - wght_max_int

            # all data are in NCHW
            output_size = conv2d_output_shape(
                (input.size()[2], input.size()[3]),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                pad=self.padding,
                stride=self.stride,
            )

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride
        )
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = HUBLinearFunction.apply(
            input_reshape,
            weight,
            None,
            self.rshift_input,
            self.rshift_wght,
            self.rshift_output,
            self.cycle,
            self.wght_map,
        )
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


class FxpConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
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
        keep_res="input",  # keep the resolution of input/output
        more_res="input",  # assign more resolution to input/weight
        rounding="round",
    ):
        super(FxpConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        assert groups == 1, "Supported group number is 1."
        assert padding_mode == "zeros", "Supported padding_mode number is 'zeros'."

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight

        if bias and (binary_bias is not None):
            self.bias.data = binary_bias

        # bitwidth of abs
        if isinstance(bitwidth, tuple):
            self.bw_input, self.bw_wght = (bitwidth[0] - 1, bitwidth[1] - 1)
        else:
            if keep_res == "input":
                self.bw_input, self.bw_wght = (bitwidth - 1, bitwidth - 1)
            elif keep_res == "output":
                if bitwidth % 2 == 0:
                    self.bw_input, self.bw_wght = (
                        int(bitwidth / 2 - 1),
                        int(bitwidth / 2 - 1),
                    )
                else:
                    if more_res == "input":
                        self.bw_input, self.bw_wght = (
                            int((bitwidth + 1) / 2 - 1),
                            int((bitwidth - 1) / 2 - 1),
                        )
                    elif more_res == "weight":
                        self.bw_input, self.bw_wght = (
                            int((bitwidth - 1) / 2 - 1),
                            int((bitwidth + 1) / 2 - 1),
                        )
                    else:
                        raise ValueError(
                            "more_res should be either 'input' or 'weight' when bitwidth is not a tuple and keep_res is 'output'."
                        )
            else:
                raise ValueError(
                    "keep_res should be either 'input' or 'output' when bitwidth is not a tuple."
                )

        # max abs value
        self.max_abs_input = 2**self.bw_input
        self.max_abs_wght = 2**self.bw_wght

        # rounding mode
        self.rounding = rounding

        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None

    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            if self.rshift_input is None:
                input_max_int = input.abs().max().log2()
                if self.rounding == "round":
                    input_max_int = input_max_int.round()
                elif self.rounding == "floor":
                    input_max_int = input_max_int.floor()
                elif self.rounding == "ceil":
                    input_max_int = input_max_int.ceil()
                self.rshift_input = input_max_int - self.bw_input

            if self.rshift_wght is None:
                wght_max_int = self.weight.abs().max().log2()
                if self.rounding == "round":
                    wght_max_int = wght_max_int.round()
                elif self.rounding == "floor":
                    wght_max_int = wght_max_int.floor()
                elif self.rounding == "ceil":
                    wght_max_int = wght_max_int.ceil()
                self.rshift_wght = wght_max_int - self.bw_wght

            if self.rshift_output is None:
                self.rshift_output = 0 - self.rshift_input - self.rshift_wght

            # all data are in NCHW
            output_size = conv2d_output_shape(
                (input.size()[2], input.size()[3]),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                pad=self.padding,
                stride=self.stride,
            )

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride
        )
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = FxpLinearFunction.apply(
            input_reshape,
            weight,
            None,
            self.rshift_input,
            self.rshift_wght,
            self.rshift_output,
            self.max_abs_input,
            self.max_abs_wght,
        )
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


class FSULinear(torch.nn.Module):
    """
    This module is the fully connected layer,
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        mode="bipolar",
        scaled=True,
        scale=None,
        depth=12,
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
        bias=True,
    ):
        super(FSULinear, self).__init__()

        self.stype = stype
        self.PC = FSULinearPC(
            in_features,
            out_features,
            bias=bias,
            binary_weight=binary_weight,
            binary_bias=binary_bias,
            bitwidth=bitwidth,
            mode=mode,
            btype=btype,
            rtype=rtype,
            stype=stype,
        )
        if scaled is True:
            if scale is None:
                scale_add = in_features + bias
            else:
                scale_add = scale
        else:
            scale_add = 1.0
        self.ACC = FSUAdd(
            mode=mode,
            scaled=scaled,
            scale=scale_add,
            dim=0,
            depth=depth,
            entry=in_features + bias,
            stype=stype,
        )

    @autocast()
    def forward(self, input, scale=None, entry=None):
        pc = self.PC(input)
        output = self.ACC(pc.unsqueeze(0), scale, entry)
        return output.type(self.stype)


class FSULinearPC(torch.nn.Linear):
    """
    This module is the parallel counter result of FSULinear before generating the bitstreams.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        mode="bipolar",
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(FSULinearPC, self).__init__(in_features, out_features, bias=bias)
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        self.mode = mode

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the linear weight and bias
        if binary_weight is not None:
            assert (binary_weight.size()[0], binary_weight.size()[1]) == (
                out_features,
                in_features,
            ), "Incorrect weight shape."
            self.weight.data = SourceGen(
                binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        if bias and (binary_bias is not None):
            assert binary_bias.size()[0] == out_features, "Incorrect bias shape."
            self.bias.data = SourceGen(
                binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        # define the kernel linear
        self.weight_bsg = BSGen(self.weight, self.rng, stype=stype)
        self.weight_rng_idx = torch.nn.Parameter(
            torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
        ).unsqueeze(0)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(
                torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(self.weight, self.rng, stype=stype)
            self.weight_rng_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
            ).unsqueeze(0)

    def FSULinear_PC(self, input):
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        weight_bs = self.weight_bsg(self.weight_rng_idx).type(torch.float)
        if weight_bs.size()[0] != batch:
            weight_bs = torch.cat(batch * [weight_bs], 0)
            self.weight_rng_idx = torch.cat(batch * [self.weight_rng_idx], 0)
        torch.add(
            self.weight_rng_idx,
            input.unsqueeze(1).type(torch.long),
            out=self.weight_rng_idx,
        )

        kernel_out = torch.empty(0, device=input.device)
        torch.matmul(
            input.unsqueeze(1).type(torch.float),
            weight_bs.transpose(1, 2),
            out=kernel_out,
        )
        kernel_out.squeeze_(1)

        if self.has_bias is True:
            bias_bs = self.bias_bsg(self.bias_rng_idx).type(torch.float)
            self.bias_rng_idx.add_(1)
            kernel_out += bias_bs.unsqueeze(0).expand_as(kernel_out)

        if self.mode == "unipolar":
            return kernel_out

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
                1 - input.unsqueeze(1).type(torch.long),
                out=self.weight_rng_idx_inv,
            )

            kernel_out_inv = torch.empty(0, device=input.device)
            torch.matmul(
                1 - input.unsqueeze(1).type(torch.float),
                weight_bs_inv.transpose(1, 2),
                out=kernel_out_inv,
            )
            kernel_out_inv.squeeze_(1)

            return kernel_out + kernel_out_inv

    @autocast()
    def forward(self, input):
        return self.FSULinear_PC(input).type(self.stype)


class FSULinearuGEMM(torch.nn.Linear):
    """
    This module is the fully connected layer using uGEMM add implementation
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        mode="bipolar",
        scaled=True,
        btype=torch.float,
        rtype=torch.float,
        stype=torch.float,
    ):
        super(FSULinearuGEMM, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        # upper bound for accumulation counter in scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((in_features - 1) / 2)
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("FSULinear mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = SourceGen(
                binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        if bias and (binary_bias is not None):
            self.bias.data = SourceGen(
                binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype
            )()

        # define the kernel linear
        self.weight_bsg = BSGen(self.weight, self.rng, stype=stype)
        self.weight_rng_idx = torch.nn.Parameter(
            torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
        ).unsqueeze(0)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(
                torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(self.weight, self.rng, stype=stype)
            self.weight_rng_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False
            ).unsqueeze(0)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

    def FSUKernel_accumulation(self, input):
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        weight_bs = self.weight_bsg(self.weight_rng_idx).type(torch.float)
        if weight_bs.size()[0] != batch:
            weight_bs = torch.cat(batch * [weight_bs], 0)
            self.weight_rng_idx = torch.cat(batch * [self.weight_rng_idx], 0)
        torch.add(
            self.weight_rng_idx,
            input.unsqueeze(1).type(torch.long),
            out=self.weight_rng_idx,
        )

        kernel_out = torch.empty(0, device=input.device)
        torch.matmul(
            input.unsqueeze(1).type(torch.float),
            weight_bs.transpose(1, 2),
            out=kernel_out,
        )
        kernel_out.squeeze_(1)

        if self.has_bias is True:
            bias_bs = self.bias_bsg(self.bias_rng_idx).type(torch.float)
            self.bias_rng_idx.add_(1)
            kernel_out += bias_bs.unsqueeze(0).expand_as(kernel_out)

        if self.mode == "unipolar":
            return kernel_out

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
                1 - input.unsqueeze(1).type(torch.long),
                out=self.weight_rng_idx_inv,
            )

            kernel_out_inv = torch.empty(0, device=input.device)
            torch.matmul(
                1 - input.unsqueeze(1).type(torch.float),
                weight_bs_inv.transpose(1, 2),
                out=kernel_out_inv,
            )
            kernel_out_inv.squeeze_(1)

            return kernel_out + kernel_out_inv

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


class GainesLinear1(torch.nn.Module):
    """
    gMUL + gADD
    this module is the fully connected layer,
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        bias=True,
        mode="bipolar",
        scaled=True,
        depth=8,
        rng_idx=1,
    ):
        super(GainesLinear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # upper bound for accumulation counter in non-scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((in_features - 1) / 2)
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("GainesLinear1 mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNGMulti(self.bitwidth, in_features, "Sobol")()
        self.rng_bias = RNG(self.bitwidth, in_features + 1, "Sobol")()

        # define the convolution weight and bias
        self.buf_wght = SourceGen(binary_weight, bitwidth=self.bitwidth, mode=mode)()
        if self.has_bias is True:
            self.buf_bias = SourceGen(binary_bias, bitwidth=self.bitwidth, mode=mode)()

        # define the kernel linear
        self.kernel = torch.nn.Linear(
            self.in_features, self.out_features, bias=self.has_bias
        )
        self.buf_wght_bs = BSGenMulti(self.buf_wght, self.rng, dim=0)
        self.rng_wght_idx = torch.nn.Parameter(
            torch.zeros_like(self.kernel.weight, dtype=torch.long), requires_grad=False
        )
        if self.has_bias is True:
            self.buf_bias_bs = BSGen(self.buf_bias, self.rng_bias)
            self.rng_bias_idx = torch.nn.Parameter(
                torch.zeros_like(self.kernel.bias, dtype=torch.long),
                requires_grad=False,
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.kernel_inv = torch.nn.Linear(
                self.in_features, self.out_features, bias=False
            )

        self.parallel_cnt = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.long), requires_grad=False
        )

        if self.scaled is True:
            self.rng_scale = RNG(
                round(math.log2(self.acc_bound.item())), (rng_idx + 5) % 1111, "Sobol"
            )()
            self.rng_scale_idx = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long), requires_grad=False
            )
        elif self.scaled is False:
            self.input_cnt = self.acc_bound.item()
            self.max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2**depth - 1),
                requires_grad=False,
            )
            self.half_max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )
            self.cnt = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )

    def GainesKernel_accumulation(self, input):
        # generate weight and bias bits for current cycle
        self.kernel.weight.data = self.buf_wght_bs(self.rng_wght_idx).type(torch.float)
        self.rng_wght_idx.add_(1)
        if self.has_bias is True:
            self.kernel.bias.data = self.buf_bias_bs(self.rng_bias_idx).type(
                torch.float
            )
            self.rng_bias_idx.add_(1)

        kernel_out = self.kernel(input.type(torch.float))

        if self.mode == "unipolar":
            return kernel_out

        if self.mode == "bipolar":
            self.kernel_inv.weight.data = 1 - self.kernel.weight.data
            kernel_out_inv = self.kernel_inv(1 - input.type(torch.float))
            return kernel_out + kernel_out_inv

    def forward(self, input):
        self.parallel_cnt.data = self.GainesKernel_accumulation(input).type(torch.long)

        if self.scaled is True:
            output = torch.ge(
                self.parallel_cnt.data,
                self.rng_scale[self.rng_scale_idx % len(self.rng_scale)],
            )
            self.rng_scale_idx.add_(1)
        else:
            if self.mode == "unipolar":
                output = torch.gt(self.parallel_cnt, 0)
            elif self.mode == "bipolar":
                self.parallel_cnt.mul_(2).sub_(self.input_cnt)
                self.cnt.data = self.cnt.add(self.parallel_cnt).clamp(
                    0, self.max.item()
                )
                output = torch.gt(self.cnt, self.half_max)

        return output.type(torch.int8)


class GainesLinear2(torch.nn.Module):
    """
    gMUL + uADD
    this module is the fully connected layer,
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        bias=True,
        mode="bipolar",
        scaled=True,
        depth=8,
        rng_idx=1,
    ):
        super(GainesLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # upper bound for accumulation counter in non-scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((in_features - 1) / 2)
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("GainesLinear2 mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNGMulti(self.bitwidth, in_features, "Sobol")()
        self.rng_bias = RNG(self.bitwidth, in_features + 1, "Sobol")()

        # define the convolution weight and bias
        self.buf_wght = SourceGen(binary_weight, bitwidth=self.bitwidth, mode=mode)()
        if self.has_bias is True:
            self.buf_bias = SourceGen(binary_bias, bitwidth=self.bitwidth, mode=mode)()

        # define the kernel linear
        self.kernel = torch.nn.Linear(
            self.in_features, self.out_features, bias=self.has_bias
        )
        self.buf_wght_bs = BSGenMulti(self.buf_wght, self.rng, dim=0)
        self.rng_wght_idx = torch.nn.Parameter(
            torch.zeros_like(self.kernel.weight, dtype=torch.long), requires_grad=False
        )
        if self.has_bias is True:
            self.buf_bias_bs = BSGen(self.buf_bias, self.rng_bias)
            self.rng_bias_idx = torch.nn.Parameter(
                torch.zeros_like(self.kernel.bias, dtype=torch.long),
                requires_grad=False,
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.kernel_inv = torch.nn.Linear(
                self.in_features, self.out_features, bias=False
            )

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

    def GainesKernel_accumulation(self, input):
        # generate weight and bias bits for current cycle
        self.kernel.weight.data = self.buf_wght_bs(self.rng_wght_idx).type(torch.float)
        self.rng_wght_idx.add_(1)
        if self.has_bias is True:
            self.kernel.bias.data = self.buf_bias_bs(self.rng_bias_idx).type(
                torch.float
            )
            self.rng_bias_idx.add_(1)

        kernel_out = self.kernel(input.type(torch.float))

        if self.mode == "unipolar":
            return kernel_out

        if self.mode == "bipolar":
            self.kernel_inv.weight.data = 1 - self.kernel.weight.data
            kernel_out_inv = self.kernel_inv(1 - input.type(torch.float))
            return kernel_out + kernel_out_inv

    def forward(self, input):
        if self.scaled is True:
            self.accumulator.data = self.accumulator.add(
                self.GainesKernel_accumulation(input)
            )
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.data = self.accumulator.add(
                self.GainesKernel_accumulation(input)
            )
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(torch.int8)


class GainesLinear3(torch.nn.Module):
    """
    uMUL + gADD: this version will not work well, due to same rng is used in uMUL, the accumulation
    will be inaccurate.
    this module is the fully connected layer,
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        bias=True,
        mode="bipolar",
        scaled=True,
        depth=8,
        rng_idx=1,
    ):
        super(GainesLinear3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # upper bound for accumulation counter in non-scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((in_features - 1) / 2)
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("GainesLinear3 mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()

        # define the convolution weight and bias
        self.buf_wght = SourceGen(binary_weight, bitwidth=self.bitwidth, mode=mode)()
        if self.has_bias is True:
            self.buf_bias = SourceGen(binary_bias, bitwidth=self.bitwidth, mode=mode)()

        # define the kernel linear
        self.kernel = torch.nn.Linear(
            self.in_features, self.out_features, bias=self.has_bias
        )
        self.buf_wght_bs = BSGen(self.buf_wght, self.rng)
        self.rng_wght_idx = torch.nn.Parameter(
            torch.zeros_like(self.kernel.weight, dtype=torch.long), requires_grad=False
        )
        if self.has_bias is True:
            self.buf_bias_bs = BSGen(self.buf_bias, self.rng)
            self.rng_bias_idx = torch.nn.Parameter(
                torch.zeros_like(self.kernel.bias, dtype=torch.long),
                requires_grad=False,
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.kernel_inv = torch.nn.Linear(
                self.in_features, self.out_features, bias=False
            )
            self.buf_wght_bs_inv = BSGen(self.buf_wght, self.rng)
            self.rng_wght_idx_inv = torch.nn.Parameter(
                torch.zeros_like(self.kernel_inv.weight, dtype=torch.long),
                requires_grad=False,
            )

        self.parallel_cnt = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.long), requires_grad=False
        )

        if self.scaled is True:
            self.rng_scale = RNG(
                round(math.log2(self.acc_bound.item())), (rng_idx + 5) % 1111, "Sobol"
            )()
            self.rng_scale_idx = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long), requires_grad=False
            )
        elif self.scaled is False:
            self.input_cnt = self.acc_bound.item()
            self.max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2**depth - 1),
                requires_grad=False,
            )
            self.half_max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )
            self.cnt = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )

    def GainesKernel_accumulation(self, input):
        # generate weight and bias bits for current cycle
        self.kernel.weight.data = self.buf_wght_bs(self.rng_wght_idx).type(torch.float)
        self.rng_wght_idx.add_(input.type(torch.long))
        if self.has_bias is True:
            self.kernel.bias.data = self.buf_bias_bs(self.rng_bias_idx).type(
                torch.float
            )
            self.rng_bias_idx.add_(1)

        kernel_out = self.kernel(input.type(torch.float))

        if self.mode == "unipolar":
            return kernel_out

        if self.mode == "bipolar":
            self.kernel_inv.weight.data = 1 - self.buf_wght_bs_inv(
                self.rng_wght_idx_inv
            ).type(torch.float)
            self.rng_wght_idx_inv.add_(1 - input.type(torch.long))
            kernel_out_inv = self.kernel_inv(1 - input.type(torch.float))
            return kernel_out + kernel_out_inv

    def forward(self, input):
        self.parallel_cnt.data = self.GainesKernel_accumulation(input).type(torch.long)

        if self.scaled is True:
            output = torch.ge(
                self.parallel_cnt.data,
                self.rng_scale[self.rng_scale_idx % len(self.rng_scale)],
            )
            self.rng_scale_idx.add_(1)
        else:
            if self.mode == "unipolar":
                output = torch.gt(self.parallel_cnt, 0)
            elif self.mode == "bipolar":
                self.parallel_cnt.mul_(2).sub_(self.input_cnt)
                self.cnt.data = self.cnt.add(self.parallel_cnt).clamp(
                    0, self.max.item()
                )
                output = torch.gt(self.cnt, self.half_max)

        return output.type(torch.int8)


class GainesLinear4(torch.nn.Module):
    """
    gMUL + gADD,
    this module is the same as GainesLinear1, except the rng is lfsr
    this module is the fully connected layer,
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) accumulation mode
    2) unary data mode
    3) binary data width
    4) binary weight
    5) binary bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        bias=True,
        mode="bipolar",
        scaled=True,
        depth=8,
        rng_idx=1,
    ):
        super(GainesLinear4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # upper bound for accumulation counter in non-scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        if bias is True:
            self.acc_bound.add_(1)

        self.mode = mode
        self.scaled = scaled

        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((in_features - 1) / 2)
            if bias is True:
                self.offset.add_(1 / 2)
        else:
            raise ValueError("GainesLinear4 mode is not implemented.")

        # bias indication for original linear layer
        self.has_bias = bias

        # data bit width
        self.bitwidth = bitwidth

        # random_sequence from sobol RNG
        self.rng = RNGMulti(self.bitwidth, in_features, "LFSR")()
        self.rng_bias = RNG(self.bitwidth, in_features + 1, "LFSR")()

        # define the convolution weight and bias
        self.buf_wght = SourceGen(binary_weight, bitwidth=self.bitwidth, mode=mode)()
        if self.has_bias is True:
            self.buf_bias = SourceGen(binary_bias, bitwidth=self.bitwidth, mode=mode)()

        # define the kernel linear
        self.kernel = torch.nn.Linear(
            self.in_features, self.out_features, bias=self.has_bias
        )
        self.buf_wght_bs = BSGenMulti(self.buf_wght, self.rng, dim=0)
        self.rng_wght_idx = torch.nn.Parameter(
            torch.zeros_like(self.kernel.weight, dtype=torch.long), requires_grad=False
        )
        if self.has_bias is True:
            self.buf_bias_bs = BSGen(self.buf_bias, self.rng_bias)
            self.rng_bias_idx = torch.nn.Parameter(
                torch.zeros_like(self.kernel.bias, dtype=torch.long),
                requires_grad=False,
            )

        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.kernel_inv = torch.nn.Linear(
                self.in_features, self.out_features, bias=False
            )

        self.parallel_cnt = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.long), requires_grad=False
        )

        if self.scaled is True:
            self.rng_scale = RNG(
                round(math.log2(self.acc_bound.item())), (rng_idx + 5) % 1111, "LFSR"
            )()
            self.rng_scale_idx = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long), requires_grad=False
            )
        elif self.scaled is False:
            self.input_cnt = self.acc_bound.item()
            self.max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2**depth - 1),
                requires_grad=False,
            )
            self.half_max = torch.nn.Parameter(
                torch.ones(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )
            self.cnt = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.long).fill_(2 ** (depth - 1)),
                requires_grad=False,
            )

    def GainesKernel_accumulation(self, input):
        # generate weight and bias bits for current cycle
        self.kernel.weight.data = self.buf_wght_bs(self.rng_wght_idx).type(torch.float)
        self.rng_wght_idx.add_(1)
        if self.has_bias is True:
            self.kernel.bias.data = self.buf_bias_bs(self.rng_bias_idx).type(
                torch.float
            )
            self.rng_bias_idx.add_(1)

        kernel_out = self.kernel(input.type(torch.float))

        if self.mode == "unipolar":
            return kernel_out

        if self.mode == "bipolar":
            self.kernel_inv.weight.data = 1 - self.kernel.weight.data
            kernel_out_inv = self.kernel_inv(1 - input.type(torch.float))
            return kernel_out + kernel_out_inv

    def forward(self, input):
        self.parallel_cnt.data = self.GainesKernel_accumulation(input).type(torch.long)

        if self.scaled is True:
            output = torch.ge(
                self.parallel_cnt.data,
                self.rng_scale[self.rng_scale_idx % len(self.rng_scale)],
            )
            self.rng_scale_idx.add_(1)
        else:
            if self.mode == "unipolar":
                output = torch.gt(self.parallel_cnt, 0)
            elif self.mode == "bipolar":
                self.parallel_cnt.mul_(2).sub_(self.input_cnt)
                self.cnt.data = self.cnt.add(self.parallel_cnt).clamp(
                    0, self.max.item()
                )
                output = torch.gt(self.cnt, self.half_max)

        return output.type(torch.int8)


# the commented FSULinearSA and FSULinearSAFunction are cycle accurate implementations
# class FSULinearSA(torch.nn.Linear):
#     """
#     this module is the fully connected layer, with binary input and binary output
#     its API is similar to the parent class (input/output feature count, bias flag), except:
#     1) binary data scale factor
#     2) binary weight
#     3) binary bias
#     4) mac cycle
#     """
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  bias=True,
#                  binary_weight=None,
#                  binary_bias=None,
#                  input_format=(1, 3, 4),
#                  weight_format=(1, 3, 4),
#                  cycle=128):
#         super(FSULinearSA, self).__init__(in_features, out_features, bias)

#         # weight and bias
#         if binary_weight is not None:
#             self.weight.data = binary_weight

#         if bias and (binary_bias is not None):
#             self.bias.data = binary_bias

#         # input format
#         self.input_format = input_format

#         # weight format
#         self.weight_format = weight_format

#         # mac computing cycle
#         self.cycle = min(cycle, 2**(input_format[1] + input_format[2]), 2**(weight_format[1] + weight_format[2]))

#         # bitwidth of rng
#         self.bitwidth = (self.cycle - 1).bit_length()
#         assert cycle == 2**self.bitwidth, "Input cycle count is not power of 2."

#         # random_sequence from sobol RNG
#         self.rng = RNG(self.bitwidth, 1, "Sobol")()

#     @autocast()
#     def forward(self, input):
#         # See the autograd section for explanation of what happens here.
#         return FSULinearSAFunction.apply(input, self.weight, self.bias, self.input_format, self.weight_format, self.cycle, self.bitwidth, self.rng)


# # Inherit from Function
# class FSULinearSAFunction(torch.autograd.Function):

#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, input, weight, bias=None,
#                 input_format=(1, 3, 4),
#                 weight_format=(1, 3, 4),
#                 cycle=128,
#                 bitwidth=7,
#                 rng=None):
#         ctx.save_for_backward(input, weight, bias)

#         # first dim should always be batch
#         batch = input.size()[0]

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         # input bsg prepare
#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         # scale input to range [0, 1]
#         scaled_abs_input = torch.zeros(0, device=input.device)
#         torch.abs((input >> input_format[1]), out=scaled_abs_input)

#         # generate src, valued 0~2^bitwidth-1
#         buf_input = torch.zeros(0, device=input.device)
#         buf_input = scaled_abs_input << bitwidth
#         buf_input.unsqueeze_(1)

#         # rng index
#         rng_input_idx = torch.zeros(1, dtype=torch.long, device=input.device)

#         # sign for accumulation
#         sign_input = torch.zeros(0, device=input.device)
#         torch.sign(input, out=sign_input)
#         sign_input.unsqueeze_(1)

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         # weight bsg prepare
#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         # scale weight to range [0, 1]
#         scaled_abs_wght = torch.zeros(0, device=weight.device)
#         torch.abs((weight >> weight_format[1]), out=scaled_abs_wght)

#         # generate src with batch, valued 0~2^bitwidth-1
#         buf_wght_no_batch = torch.zeros(0, device=weight.device)
#         buf_wght_no_batch = scaled_abs_wght << bitwidth
#         buf_wght_no_batch.unsqueeze_(0)
#         buf_wght = torch.zeros(0, device=weight.device)
#         torch.cat(batch*[buf_wght_no_batch], 0, out=buf_wght)

#         # rng index
#         rng_wght_idx = torch.zeros(0, device=weight.device)
#         torch.zeros(buf_wght.size(), out=rng_wght_idx, device=weight.device)

#         # sign for accumulation
#         sign_wght_no_batch = torch.zeros(0, device=weight.device)
#         torch.sign(weight, out=sign_wght_no_batch)
#         sign_wght_no_batch.unsqueeze_(0)
#         sign_wght = torch.zeros(0, device=weight.device)
#         torch.cat(batch*[sign_wght_no_batch], 0, out=sign_wght)

#         mm_out = torch.zeros(0, device=input.device)
#         output = torch.zeros(input.matmul(weight.t()).size(), device=input.device).unsqueeze_(1)

#         input_b_unsign = torch.zeros(0, device=input.device)
#         input_b = torch.zeros(0, device=input.device)
#         wght_b_unsign = torch.zeros(0, device=weight.device)
#         wght_b = torch.zeros(0, device=weight.device)
#         wght_rand = torch.zeros(0, device=weight.device)

#         for c in range(cycle):
#             rng_input_idx.fill_(c)
#             torch.gt(buf_input, rng[rng_input_idx], out=input_b_unsign)
#             torch.mul(input_b_unsign.type(torch.float), sign_input, out=input_b)

#             torch.gt(buf_wght, rng[rng_wght_idx.type(torch.long)], out=wght_b_unsign)
#             torch.mul(wght_b_unsign.type(torch.float), sign_wght, out=wght_b)
#             torch.add(rng_wght_idx, input_b_unsign.type(torch.float), out=rng_wght_idx)

#             torch.baddbmm(output, input_b, wght_b.transpose(1, 2), out=output)

#         output = (((output >> bitwidth) << input_format[1]) << weight_format[1]).squeeze_(1)

#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None

#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.matmul(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().matmul(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)

#         return grad_input, grad_weight, grad_bias, None, None, None, None, None


# the HUBLinear and HUBLinearFunction are parallel implementations
class HUBLinear(torch.nn.Linear):
    """
    this module is the fully connected layer, with binary input and binary output
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) binary data scale factor
    2) binary weight
    3) binary bias
    4) mac cycle
    This cycle is the mac cycle using unipolar umul, i.e., half the bipolar umul.
    As such, cycle = 2 ^ (bitwidth - 1).
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        binary_weight=None,
        binary_bias=None,
        rng="Sobol",
        cycle=128,
        rounding="round",
    ):
        super(HUBLinear, self).__init__(in_features, out_features, bias)

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight

        if bias and (binary_bias is not None):
            self.bias.data = binary_bias

        # mac computing cycle
        self.cycle = cycle

        # bitwidth of rng
        self.bitwidth = (self.cycle - 1).bit_length()

        # random_sequence from sobol RNG
        self.irng = RNG(self.bitwidth, 1, rng)()
        self.wrng = RNG(self.bitwidth, 1, "Sobol")()

        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the rng
        self.input_map = torch.nn.Parameter(torch.empty(cycle), requires_grad=False)
        input_val_cycle = torch.empty(0)
        torch.cat(
            cycle * [torch.arange(cycle, dtype=torch.float).unsqueeze(1)],
            1,
            out=input_val_cycle,
        )
        input_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.irng.unsqueeze(0), out=input_bit_cycle)
        self.input_map.data = torch.sum(input_bit_cycle, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the rng, under a specific input and weight
        self.wght_map = torch.nn.Parameter(
            torch.empty(cycle, cycle), requires_grad=False
        )
        wght_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.wrng.unsqueeze(0), out=wght_bit_cycle)
        for c in range(cycle):
            self.wght_map.data[c] = torch.sum(
                wght_bit_cycle[:, 0 : self.input_map.data[c]], 1
            ).squeeze_()

        # rounding mode
        self.rounding = rounding

        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None

    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            input_max_int = input.abs().max().log2()
            wght_max_int = self.weight.abs().max().log2()
            if self.rounding == "round":
                input_max_int = input_max_int.round()
                wght_max_int = wght_max_int.round()
            elif self.rounding == "floor":
                input_max_int = input_max_int.floor()
                wght_max_int = wght_max_int.floor()
            elif self.rounding == "ceil":
                input_max_int = input_max_int.ceil()
                wght_max_int = wght_max_int.ceil()

            self.rshift_input = input_max_int - self.bitwidth
            self.rshift_wght = wght_max_int - self.bitwidth
            self.rshift_output = self.bitwidth - input_max_int - wght_max_int

        return HUBLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.rshift_input,
            self.rshift_wght,
            self.rshift_output,
            self.cycle,
            self.wght_map,
        )


# Inherit from Function
class HUBLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        rshift_input=3,
        rshift_wght=3,
        rshift_output=3,
        cycle=128,
        wght_map=None,
    ):
        ctx.save_for_backward(input, weight, bias)

        # first dim should always be batch
        batch = input.size()[0]

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # scale input to range 0~2^bitwidth-1
        buf_input = torch.empty(0, dtype=torch.long, device=input.device)
        torch.abs((input >> rshift_input).unsqueeze_(1).type(torch.long), out=buf_input)
        torch.clamp(buf_input, 0, cycle - 1, out=buf_input)

        # actual input: its sign
        act_input = torch.empty(0, device=input.device)
        torch.sign(input, out=act_input)
        act_input.unsqueeze_(1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # scale weight with batch to range 0~2^bitwidth-1
        buf_wght_no_batch = torch.empty(0, dtype=torch.long, device=weight.device)
        torch.abs(
            (weight >> rshift_wght).unsqueeze_(0).type(torch.long),
            out=buf_wght_no_batch,
        )
        torch.clamp(buf_wght_no_batch, 0, cycle - 1, out=buf_wght_no_batch)
        buf_wght = torch.empty(0, dtype=torch.long, device=weight.device)
        torch.cat(batch * [buf_wght_no_batch], 0, out=buf_wght)

        # get actual weight for calculation
        sign_wght_no_batch = torch.empty(0, device=weight.device)
        torch.sign(weight, out=sign_wght_no_batch)
        sign_wght_no_batch.unsqueeze_(0)
        act_wght = torch.empty(0, device=weight.device)
        torch.cat(batch * [sign_wght_no_batch], 0, out=act_wght)
        torch.mul(wght_map[buf_input, buf_wght], act_wght, out=act_wght)

        output = torch.empty(0, device=weight.device)
        torch.matmul(act_input, act_wght.transpose(1, 2), out=output)

        output = (output >> rshift_output).squeeze_(1)

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class FxpLinear(torch.nn.Linear):
    """
    this module is the fully connected layer, with binary input and binary output
    its API is similar to the parent class (input/output feature count, bias flag), except:
    1) binary data scale factor
    2) binary weight
    3) binary bias
    4) mac cycle
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        binary_weight=None,
        binary_bias=None,
        bitwidth=8,
        keep_res="input",  # keep the resolution of input/output
        more_res="input",  # assign more resolution to input/weight
        rounding="round",
    ):
        super(FxpLinear, self).__init__(in_features, out_features, bias)

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight

        if bias and (binary_bias is not None):
            self.bias.data = binary_bias

        # bitwidth of abs
        if isinstance(bitwidth, tuple):
            self.bw_input, self.bw_wght = (bitwidth[0] - 1, bitwidth[1] - 1)
        else:
            if keep_res == "input":
                self.bw_input, self.bw_wght = (bitwidth - 1, bitwidth - 1)
            elif keep_res == "output":
                if bitwidth % 2 == 0:
                    self.bw_input, self.bw_wght = (
                        int(bitwidth / 2 - 1),
                        int(bitwidth / 2 - 1),
                    )
                else:
                    if more_res == "input":
                        self.bw_input, self.bw_wght = (
                            int((bitwidth + 1) / 2 - 1),
                            int((bitwidth - 1) / 2 - 1),
                        )
                    elif more_res == "weight":
                        self.bw_input, self.bw_wght = (
                            int((bitwidth - 1) / 2 - 1),
                            int((bitwidth + 1) / 2 - 1),
                        )
                    else:
                        raise ValueError(
                            "more_res should be either 'input' or 'weight' when bitwidth is not a tuple and keep_res is 'output'."
                        )
            else:
                raise ValueError(
                    "keep_res should be either 'input' or 'output' when bitwidth is not a tuple."
                )

        # max abs value
        self.max_abs_input = 2**self.bw_input
        self.max_abs_wght = 2**self.bw_wght

        # rounding mode
        self.rounding = rounding

        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None

    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            if self.rshift_input is None:
                input_max_int = input.abs().max().log2()
                if self.rounding == "round":
                    input_max_int = input_max_int.round()
                elif self.rounding == "floor":
                    input_max_int = input_max_int.floor()
                elif self.rounding == "ceil":
                    input_max_int = input_max_int.ceil()
                self.rshift_input = input_max_int - self.bw_input

            if self.rshift_wght is None:
                wght_max_int = self.weight.abs().max().log2()
                if self.rounding == "round":
                    wght_max_int = wght_max_int.round()
                elif self.rounding == "floor":
                    wght_max_int = wght_max_int.floor()
                elif self.rounding == "ceil":
                    wght_max_int = wght_max_int.ceil()
                self.rshift_wght = wght_max_int - self.bw_wght

            if self.rshift_output is None:
                self.rshift_output = 0 - self.rshift_input - self.rshift_wght

        return FxpLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.rshift_input,
            self.rshift_wght,
            self.rshift_output,
            self.max_abs_input,
            self.max_abs_wght,
        )


# Inherit from Function
class FxpLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        rshift_input=3,
        rshift_wght=3,
        rshift_output=3,
        max_abs_input=128,
        max_abs_wght=128,
    ):
        ctx.save_for_backward(input, weight, bias)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        bot_input = 0 - max_abs_input
        top_input = max_abs_input - 1
        input_round = torch.empty(0, device=input.device)
        torch.round(input >> rshift_input, out=input_round)
        torch.clamp(input_round.unsqueeze_(1), bot_input, top_input, out=input_round)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # round input to (bot, top)
        bot_wght = 0 - max_abs_wght
        top_wght = max_abs_wght - 1
        wght_round = torch.empty(0, device=input.device)
        torch.round(weight >> rshift_wght, out=wght_round)
        torch.clamp(wght_round.unsqueeze_(0), bot_wght, top_wght, out=wght_round)

        output = torch.empty(0, device=weight.device)
        torch.matmul(input_round, wght_round.transpose(1, 2), out=output)
        output = (output >> rshift_output).squeeze_(1)

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
