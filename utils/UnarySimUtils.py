import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

import torch
import numpy as np
from pylfsr import LFSR


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


def convtransp2d_output_shape(
    h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0
):
    h_w, kernel_size, stride, pad, dilation, out_pad = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
        num2tuple(out_pad),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (
        (h_w[0] - 1) * stride[0]
        - sum(pad[0])
        + dilation[0] * (kernel_size[0] - 1)
        + out_pad[0]
        + 1
    )
    w = (
        (h_w[1] - 1) * stride[1]
        - sum(pad[1])
        + dilation[1] * (kernel_size[1] - 1)
        + out_pad[1]
        + 1
    )

    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = (
        num2tuple(h_w_in),
        num2tuple(h_w_out),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(dilation),
    )

    p_h = (
        (h_w_out[0] - 1) * stride[0]
        - h_w_in[0]
        + dilation[0] * (kernel_size[0] - 1)
        + 1
    )
    p_w = (
        (h_w_out[1] - 1) * stride[1]
        - h_w_in[1]
        + dilation[1] * (kernel_size[1] - 1)
        + 1
    )

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (
        math.floor(p_w / 2),
        math.ceil(p_w / 2),
    )


def convtransp2d_get_padding(
    h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0
):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = (
        num2tuple(h_w_in),
        num2tuple(h_w_out),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(dilation),
        num2tuple(out_pad),
    )

    p_h = (
        -(
            h_w_out[0]
            - 1
            - out_pad[0]
            - dilation[0] * (kernel_size[0] - 1)
            - (h_w_in[0] - 1) * stride[0]
        )
        / 2
    )
    p_w = (
        -(
            h_w_out[1]
            - 1
            - out_pad[1]
            - dilation[1] * (kernel_size[1] - 1)
            - (h_w_in[1] - 1) * stride[1]
        )
        / 2
    )

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (
        math.floor(p_w / 2),
        math.ceil(p_w / 2),
    )


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if torch.sum(cond):
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones_like(t), mean=mean, std=std), t
            )
        else:
            break
    return t


def tensor_unary_outlier(tensor, name="tensor"):
    min = tensor.min().item()
    max = tensor.max().item()
    outlier = torch.sum(torch.gt(tensor, 1)) + torch.sum(torch.lt(tensor, -1))
    outlier_ratio = outlier / torch.prod(torch.tensor(tensor.size()))
    print(
        "{:30s}".format(name)
        + ": min:"
        + "{:12f}".format(min)
        + "; max:"
        + "{:12f}".format(max)
        + "; outlier:"
        + "{:12f} %".format(outlier_ratio * 100)
    )


def progerror_report(
    progerror, name="progerror", report_value=False, report_relative=False
):
    if report_value:
        min = progerror.in_value.min().item()
        max = progerror.in_value.max().item()
        std, mean = torch.std_mean(progerror()[0])
        print(
            "{:30s}".format(name)
            + ", Binary   Value range,"
            + "{:12f}".format(min)
            + ", {:12f}".format(max)
            + ", std,"
            + "{:12f}".format(std)
            + ", mean,"
            + "{:12f}".format(mean)
        )

        min = progerror()[0].min().item()
        max = progerror()[0].max().item()
        std, mean = torch.std_mean(progerror()[0])
        print(
            "{:30s}".format(name)
            + ", Unary    Value range,"
            + "{:12f}".format(min)
            + ", {:12f}".format(max)
            + ", std,"
            + "{:12f}".format(std)
            + ", mean,"
            + "{:12f}".format(mean)
        )

    min = progerror()[1].min().item()
    max = progerror()[1].max().item()
    rmse = torch.sqrt(torch.mean(torch.square(progerror()[1])))
    std, mean = torch.std_mean(progerror()[1])
    print(
        "{:30s}".format(name)
        + ", Absolute Error range,"
        + "{:12f}".format(min)
        + ", {:12f}".format(max)
        + ", std,"
        + "{:12f}".format(std)
        + ", mean,"
        + "{:12f}".format(mean)
        + ", rmse,"
        + "{:12f}".format(rmse)
    )

    if report_relative:
        relative_error = torch.nan_to_num(progerror()[1] / progerror()[0])
        min = relative_error.min().item()
        max = relative_error.max().item()
        rmse = torch.sqrt(torch.mean(torch.square(relative_error)))
        std, mean = torch.std_mean(relative_error)
        print(
            "{:30s}".format(name)
            + ", Relative Error range,"
            + "{:12f}".format(min)
            + ", {:12f}".format(max)
            + ", std,"
            + "{:12f}".format(std)
            + ", mean,"
            + "{:12f}".format(mean)
            + ", rmse,"
            + "{:12f}".format(rmse)
        )


class RoundingNoGrad(torch.autograd.Function):
    """
    RoundingNoGrad is a rounding operation which bypasses the input gradient to output directly.
    Original round()/floor()/ceil() opertions have a gradient of 0 everywhere, which is not useful
    when doing approximate computing.
    This is something like the straight-through estimator (STE) for quantization-aware training.
    Code is taken from RAVEN (https://github.com/diwu1990/RAVEN/blob/master/pe/appr_utils.py)
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input


class Round(torch.nn.Module):
    """
    Round is an operation to convert data to format (1, intwidth, fracwidth).
    """

    def __init__(self, intwidth=3, fracwidth=4) -> None:
        super(Round, self).__init__()
        self.intwidth = intwidth
        self.fracwidth = fracwidth
        self.max_val = 2 ** (intwidth + fracwidth) - 1
        self.min_val = 0 - (2 ** (intwidth + fracwidth))

    def forward(self, input) -> Tensor:
        if input is None:
            return None
        else:
            return (
                RoundingNoGrad.apply(input << self.fracwidth).clamp(
                    self.min_val, self.max_val
                )
                >> self.fracwidth
            )


class NN_SC_Weight_Clipper(object):
    """
    This is a clipper for weights and bias of neural networks
    """

    def __init__(self, frequency=1, mode="bipolar", method="clip", bitwidth=8):
        self.frequency = frequency
        # "unipolar" or "bipolar"
        self.mode = mode
        # "clip" or "norm"
        self.method = method
        self.scale = 2**bitwidth

    def __call__(self, module):
        # filter the variables to get the ones you want
        if self.frequency > 1:
            self.method = "clip"
        else:
            self.method = "norm"

        if hasattr(module, "weight"):
            w = module.weight.data
            self.clipping(w)

        if hasattr(module, "bias"):
            w = module.bias.data
            self.clipping(w)

        self.frequency = self.frequency + 1

    def clipping(self, w):
        if self.mode == "unipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)).mul_(
                    self.scale
                ).round_().clamp_(0.0, self.scale).div_(self.scale)
            elif self.method == "clip":
                w.clamp_(0.0, 1.0).mul_(self.scale).round_().clamp_(
                    0.0, self.scale
                ).div_(self.scale)
            else:
                raise TypeError(
                    "unknown method type '{}' in SC_Weight, should be 'clip' or 'norm'".format(
                        self.method
                    )
                )
        elif self.mode == "bipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)).mul_(2).sub_(
                    1
                ).mul_(self.scale / 2).round_().clamp_(
                    -self.scale / 2, self.scale / 2
                ).div_(
                    self.scale / 2
                )
            elif self.method == "clip":
                w.clamp_(-1.0, 1.0).mul_(self.scale / 2).round_().clamp_(
                    -self.scale / 2, self.scale / 2
                ).div_(self.scale / 2)
            else:
                raise TypeError(
                    "unknown method type '{}' in SC_Weight, should be 'clip' or 'norm'".format(
                        self.method
                    )
                )
        else:
            raise TypeError(
                "unknown mode type '{}' in SC_Weight, should be 'unipolar' or 'bipolar'".format(
                    self.mode
                )
            )


def get_lfsr_seq(bitwidth=8):
    polylist = LFSR().get_fpolyList(m=bitwidth)
    poly = polylist[np.random.randint(0, len(polylist), 1)[0]]
    L = LFSR(fpoly=poly, initstate="random")
    lfsr_seq = []
    for i in range(2**bitwidth):
        value = 0
        for j in range(bitwidth):
            value = value + L.state[j] * 2 ** (bitwidth - 1 - j)
        lfsr_seq.append(value)
        L.next()
    return lfsr_seq


def get_sysrand_seq(bitwidth=8):
    return torch.randperm(2**bitwidth)


class RNG(torch.nn.Module):
    """
    Random number generator to generate one random sequence, returns a tensor of size [2**bitwidth]
    returns a torch.nn.Parameter
    """

    def __init__(self, bitwidth=8, dim=1, rng="Sobol", rtype=torch.float):
        super(RNG, self).__init__()
        self.dim = dim
        self.rng = rng
        self.seq_len = pow(2, bitwidth)
        self.rng_seq = torch.nn.Parameter(
            torch.Tensor(1, self.seq_len), requires_grad=False
        )
        self.rtype = rtype
        if self.rng == "Sobol":
            # get the requested dimension of sobol random number
            self.rng_seq.data = (
                torch.quasirandom.SobolEngine(self.dim)
                .draw(self.seq_len)[:, dim - 1]
                .view(self.seq_len)
                .mul_(self.seq_len)
            )
        elif self.rng == "Race":
            self.rng_seq.data = torch.tensor(
                [x / self.seq_len for x in range(self.seq_len)]
            ).mul_(self.seq_len)
        elif self.rng == "LFSR":
            lfsr_seq = get_lfsr_seq(bitwidth=bitwidth)
            self.rng_seq.data = torch.tensor(lfsr_seq).type(torch.float)
        elif self.rng == "SYS":
            sysrand_seq = get_sysrand_seq(bitwidth=bitwidth)
            self.rng_seq.data = sysrand_seq.type(torch.float)
        else:
            raise ValueError("RNG rng is not implemented.")
        self.rng_seq.data = self.rng_seq.data.floor().type(self.rtype)

    def forward(self):
        return self.rng_seq


class RNGMulti(torch.nn.Module):
    """
    Random number generator to generate multiple random sequences, returns a tensor of size [dim, 2**bitwidth]
    returns a torch.nn.Parameter
    """

    def __init__(
        self, bitwidth=8, dim=1, rng="Sobol", transpose=False, rtype=torch.float
    ):
        super(RNGMulti, self).__init__()
        self.dim = dim
        self.rng = rng
        self.seq_len = pow(2, bitwidth)
        self.rng_seq = torch.nn.Parameter(
            torch.Tensor(1, self.seq_len), requires_grad=False
        )
        self.rtype = rtype
        if self.rng == "Sobol":
            # get the requested dimension of sobol random number
            self.rng_seq.data = (
                torch.quasirandom.SobolEngine(self.dim)
                .draw(self.seq_len)
                .mul_(self.seq_len)
            )
        elif self.rng == "LFSR":
            lfsr_seq = []
            for i in range(dim):
                lfsr_seq.append(get_lfsr_seq(bitwidth=bitwidth))
            self.rng_seq.data = torch.tensor(lfsr_seq).transpose(0, 1).type(torch.float)
        elif self.rng == "SYS":
            sysrand_seq = get_sysrand_seq(bitwidth=bitwidth)
            for i in range(dim - 1):
                temp_seq = get_sysrand_seq(bitwidth=bitwidth)
                sysrand_seq = torch.stack((sysrand_seq, temp_seq), dim=0)
            self.rng_seq.data = sysrand_seq.transpose(0, 1).type(torch.float)
        else:
            raise ValueError("RNG rng is not implemented.")
        if transpose is True:
            self.rng_seq.data = self.rng_seq.data.transpose(0, 1)
        self.rng_seq.data = self.rng_seq.data.floor().type(self.rtype)

    def forward(self):
        return self.rng_seq


class RawScale(torch.nn.Module):
    """
    Scale raw data to source data in unary computing, which meets bipolar/unipolar requirements.
    input percentile should be a number in range (0, 100].
    returns a torch.nn.Parameter
    """

    def __init__(self, raw, mode="bipolar", percentile=100):
        super(RawScale, self).__init__()
        self.raw = raw
        self.mode = mode

        # to do: add the percentile based scaling
        self.percentile_down = (100 - percentile) / 2
        self.percentile_up = 100 - self.percentile_down
        self.clamp_min = np.percentile(raw.cpu(), self.percentile_down)
        self.clamp_max = np.percentile(raw.cpu(), self.percentile_up)

        self.source = torch.nn.Parameter(torch.Tensor(raw.size()), requires_grad=False)
        self.source.data = raw.clamp(self.clamp_min, self.clamp_max)

    def forward(self):
        if self.mode == "unipolar":
            self.source.data = (self.source - torch.min(self.source)) / (
                torch.max(self.source) - torch.min(self.source)
            )
        elif self.mode == "bipolar":
            self.source.data = (self.source - torch.min(self.source)) / (
                torch.max(self.source) - torch.min(self.source)
            ) * 2 - 1
        else:
            raise ValueError("RawScale mode is not implemented.")
        return self.source


class SourceGen(torch.nn.Module):
    """
    Convert source problistic data to binary integer data
    returns a torch.nn.Parameter
    """

    def __init__(self, prob, bitwidth=8, mode="bipolar", rtype=torch.float):
        super(SourceGen, self).__init__()
        self.prob = prob
        self.mode = mode
        self.rtype = rtype
        self.len = pow(2, bitwidth)
        self.binary = torch.nn.Parameter(torch.Tensor(prob.size()), requires_grad=False)
        if mode == "unipolar":
            print(self.prob.mul(self.len).dtype)
            self.binary.data = self.prob.mul(self.len).round()
        elif mode == "bipolar":
            self.binary.data = self.prob.add(1).div(2).mul(self.len).round()
        else:
            raise ValueError("SourceGen mode is not implemented.")
        self.binary.data = self.binary.type(self.rtype)

    def forward(self):
        return self.binary


class BSGen(torch.nn.Module):
    """
    Compare source data with rng_seq[rng_idx] to generate bit streams from source
    only one rng sequence is used here
    """

    def __init__(self, source, rng_seq, stype=torch.float):
        super(BSGen, self).__init__()
        self.source = source
        self.rng_seq = rng_seq
        # print("BS Source Gen Shape :", self.source.shape)
        # print("Random Number Generator shape :", self.rng_seq.shape)
        self.stype = stype

    def forward(self, rng_idx):
        # print("Random Number Shape ", self.rng_seq[rng_idx.type(torch.long)].shape)
        return torch.gt(self.source, self.rng_seq[rng_idx.type(torch.long)]).type(
            self.stype
        )


class BSGenMulti(torch.nn.Module):
    """
    Compare source data with rng_seq indexed with rng_idx to generate bit streams from source
    multiple rng sequences are used here
    this BSGenMulti shares the random number along the dim
    """

    def __init__(self, source, rng_seq, dim=0, stype=torch.float):
        super(BSGenMulti, self).__init__()
        self.source = source
        self.rng_seq = rng_seq
        self.dim = dim
        self.stype = stype

    def forward(self, rng_idx):
        return torch.gt(
            self.source, torch.gather(self.rng_seq, self.dim, rng_idx.type(torch.long))
        ).type(self.stype)


class Correlation(torch.nn.Module):
    """
    calculate the correlation between two input bit streams.
    SC correlation: "Exploiting correlation in stochastic circuit design"
    """

    def __init__(self):
        super(Correlation, self).__init__()
        self.paired_00_d = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_01_c = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_10_b = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_11_a = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.in_1_d = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def Monitor(self, in_1, in_2=None):
        if in_2 is None:
            in_2 = self.in_1_d.clone().detach()
            self.in_1_d.data = in_1.clone().detach()
        in_1_is_0 = torch.eq(in_1, 0).type(torch.float)
        in_1_is_1 = 1 - in_1_is_0
        in_2_is_0 = torch.eq(in_2, 0).type(torch.float)
        in_2_is_1 = 1 - in_2_is_0
        self.paired_00_d.data.add_(in_1_is_0 * in_2_is_0)
        self.paired_01_c.data.add_(in_1_is_0 * in_2_is_1)
        self.paired_10_b.data.add_(in_1_is_1 * in_2_is_0)
        self.paired_11_a.data.add_(in_1_is_1 * in_2_is_1)
        self.len.data.add_(1)

    def forward(self):
        ad_minus_bc = (
            self.paired_11_a * self.paired_00_d - self.paired_10_b * self.paired_01_c
        )
        ad_gt_bc = torch.gt(ad_minus_bc, 0).type(torch.float)
        ad_le_bc = 1 - ad_gt_bc
        a_plus_b = self.paired_11_a + self.paired_10_b
        a_plus_c = self.paired_11_a + self.paired_01_c
        a_minus_d = self.paired_11_a - self.paired_00_d
        all_0_tensor = torch.zeros_like(self.paired_11_a)
        all_1_tensor = torch.ones_like(self.paired_11_a)
        corr_ad_gt_bc = ad_minus_bc.div(
            torch.max(
                torch.min(a_plus_b, a_plus_c)
                .mul_(self.len)
                .sub_(a_plus_b.mul(a_plus_c)),
                all_1_tensor,
            )
        )
        corr_ad_le_bc = ad_minus_bc.div(
            torch.max(
                a_plus_b.mul(a_plus_c).sub(
                    torch.max(a_minus_d, all_0_tensor).mul_(self.len)
                ),
                all_1_tensor,
            )
        )
        return ad_gt_bc * corr_ad_gt_bc + ad_le_bc * corr_ad_le_bc


class ProgError(torch.nn.Module):
    """
    calculate progressive error based on progressive precision of input bit stream.
    progressive precision: "Fast and accurate computation using stochastic circuits"
    scale=1 indicates non-scale, scale>1 indicates scale.
    """

    def __init__(self, in_value, scale=1, mode="bipolar"):
        super(ProgError, self).__init__()
        # in_value is always binary
        # after scaling, unipolar should be within (0, 1), bipolar should be within (-1, 1).
        # therefore, clamping with (-1, 1) always works
        # print(in_value)
        # print("Scale :", scale)
        print(type(in_value))
        self.in_value = torch.clamp(in_value, -1.0, 1.0)
        self.mode = mode
        assert (
            self.mode == "unipolar" or self.mode == "bipolar"
        ), "ProgError mode is not implemented."
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.one_cnt = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_pp = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def Monitor(self, in_1):
        self.one_cnt.data = self.one_cnt.data.add(in_1.type(torch.float))
        self.len.data.add_(1)

    def forward(self):
        self.out_pp.data = self.one_cnt.div(self.len)
        if self.mode == "bipolar":
            self.out_pp.data = self.out_pp.mul(2).sub(1)
        self.err.data = self.out_pp.sub(self.in_value)
        return self.out_pp, self.err


class Stability(torch.nn.Module):
    """
    calculate the stability of one bit stream.
    stability: "uGEMM: Unary Computing Architecture for GEMM Applications"
    """

    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(Stability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = torch.nn.Parameter(
            torch.tensor([threshold]), requires_grad=False
        )
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.stable_len = torch.zeros_like(in_value)
        self.stability = torch.zeros_like(in_value)
        self.pp = ProgError(in_value, scale=1, mode=mode)

    def Monitor(self, in_1):
        self.pp.Monitor(in_1)
        self.len = self.pp.len
        _, self.err = self.pp()
        self.stable_len.add_(
            torch.gt(self.err.abs(), self.threshold)
            .type(torch.float)
            .mul_(self.len - self.stable_len)
        )

    def forward(self):
        self.stability = 1 - self.stable_len.clamp(1, self.len.item()).div(self.len)
        return self.stability


def search_best_stab_parallel_numpy(P_low_L, P_high_L, L, seach_range):
    """
    This function is used to search the best stability length, R and l_p
    All data are numpy arrays
    """
    max_stab_len = np.ones_like(P_low_L)
    max_stab_len.fill(L.item(0))
    max_stab_R = np.ones_like(P_low_L)
    max_stab_l_p = np.ones_like(P_low_L)

    for i in range(seach_range.item(0) + 1):
        p_L = np.clip(P_low_L + i, None, P_high_L)
        l_p = L / np.gcd(p_L, L)

        l_p_by_p_L_minus_P_low_L = l_p * i
        l_p_by_P_high_L_min_p_L = l_p * (P_high_L - p_L)
        l_p_by_p_L_minus_P_low_L[l_p_by_p_L_minus_P_low_L == 0] = 1
        l_p_by_P_high_L_min_p_L[l_p_by_P_high_L_min_p_L == 0] = 1

        # one more bit 0
        B_L = 0
        p_L_eq_P_low_L = (p_L == P_low_L).astype("float32")
        P_low_L_le_B_L = (P_low_L <= B_L).astype("float32")
        R_low = (
            p_L_eq_P_low_L * ((1 - P_low_L_le_B_L) * L)
            + (1 - p_L_eq_P_low_L) * (P_low_L - B_L) / l_p_by_p_L_minus_P_low_L
        )

        P_high_L_eq_p_L = (P_high_L == p_L).astype("float32")
        B_L_le_P_high_L = (B_L <= P_high_L).astype("float32")
        R_high = (
            P_high_L_eq_p_L * ((1 - B_L_le_P_high_L) * L)
            + (1 - P_high_L_eq_p_L) * (B_L - P_high_L) / l_p_by_P_high_L_min_p_L
        )

        R_0 = np.ceil(np.maximum(R_low, R_high))

        # one more bit 0
        B_L = L
        p_L_eq_P_low_L = (p_L == P_low_L).astype("float32")
        P_low_L_le_B_L = (P_low_L <= B_L).astype("float32")
        R_low = (
            p_L_eq_P_low_L * ((1 - P_low_L_le_B_L) * L)
            + (1 - p_L_eq_P_low_L) * (P_low_L - B_L) / l_p_by_p_L_minus_P_low_L
        )

        P_high_L_eq_p_L = (P_high_L == p_L).astype("float32")
        B_L_le_P_high_L = (B_L <= P_high_L).astype("float32")
        R_high = (
            P_high_L_eq_p_L * ((1 - B_L_le_P_high_L) * L)
            + (1 - P_high_L_eq_p_L) * (B_L - P_high_L) / l_p_by_P_high_L_min_p_L
        )

        R_L = np.ceil(np.maximum(R_low, R_high))

        R = np.minimum(R_0, R_L)

        R_by_l_p = R * l_p
        R_by_l_p_lt_max_stab_len = (R_by_l_p < max_stab_len).astype("float32")

        max_stab_len = R_by_l_p_lt_max_stab_len * np.maximum(R_by_l_p, 1)
        max_stab_R = R_by_l_p_lt_max_stab_len * R
        max_stab_l_p = R_by_l_p_lt_max_stab_len * l_p

    return (
        max_stab_len.astype("float32"),
        max_stab_R.astype("float32"),
        max_stab_l_p.astype("float32"),
    )


class NormStability(torch.nn.Module):
    """
    calculate the normalized value-independent stability, which is standard stability over maximum stability.
    normalized stability is acutual stability/max possible stability
    All inputs should be on CPU
    """

    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(NormStability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = torch.nn.Parameter(
            torch.tensor([threshold]), requires_grad=False
        )
        self.stability = Stability(in_value, mode=mode, threshold=threshold)
        self.min_prob = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.max_prob = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "bipolar":
            self.min_prob.data = torch.max(
                (in_value + 1) / 2 - threshold / 2, torch.zeros_like(in_value)
            )
            self.max_prob.data = torch.min(
                (in_value + 1) / 2 + threshold / 2, torch.ones_like(in_value)
            )
        else:
            self.min_prob.data = torch.max(
                in_value - threshold, torch.zeros_like(in_value)
            )
            self.max_prob.data = torch.min(
                in_value + threshold, torch.ones_like(in_value)
            )
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.in_shape = in_value.size()
        self.max_stab = torch.zeros_like(in_value)
        self.max_stab_len = torch.ones_like(in_value)
        self.max_stab_l_p = torch.ones_like(in_value)
        self.max_stab_R = torch.ones_like(in_value)

    def Monitor(self, in_1):
        self.stability.Monitor(in_1)
        self.len.data = self.stability.len.clone().detach()

    def forward(self):
        # parallel execution based on numpy speeds up by 30X
        assert self.len != 0, "Input bit stream length can't be 0."
        L = torch.pow(2, torch.ceil(torch.log2(self.len)))
        # use ceil for lower to avoid 0
        P_low_L_all = torch.floor(self.min_prob * L).clamp(0, L.item())
        # use ceil for upper to avoid the case that upper is smaller than lower, when bit stream length is small
        P_high_L_all = torch.ceil(self.max_prob * L).clamp(0, L.item())
        seach_range = (self.threshold * 2 * L + 1).type(torch.int32)

        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab_parallel_numpy(
            P_low_L_all.type(torch.int32).numpy(),
            P_high_L_all.type(torch.int32).numpy(),
            L.type(torch.int32).numpy(),
            seach_range.numpy(),
        )

        self.max_stab.data = torch.from_numpy(
            np.maximum(1 - max_stab_len / self.len.numpy(), 0)
        )
        self.max_stab_len.data = torch.from_numpy(max_stab_len)
        self.max_stab_l_p.data = torch.from_numpy(max_stab_l_p)
        self.max_stab_R.data = torch.from_numpy(max_stab_R)

        normstab = self.stability() / self.max_stab
        normstab[torch.isnan(normstab)] = 0
        # some normstab is larger than 1.0,
        # as the method based on the segmented uniform,
        # which is an approximation of the best case
        normstab.clamp_(0, 1)

        return normstab


def gen_ns_out_parallel_numpy(
    new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L
):
    """
    This function is used to generate the output for NSbuilder
    All data are numpy arrays
    """
    out_cnt_ns_eq_new_ns_len = (out_cnt_ns >= new_ns_len).astype("int32")
    out_cnt_ns_eq_L = (out_cnt_ns >= L).astype("int32")

    ns_gen = 1 - out_cnt_ns_eq_new_ns_len
    st_gen = out_cnt_ns_eq_new_ns_len

    output = (ns_gen & bs_ns) | (st_gen & bs_st)

    out_cnt_ns = out_cnt_ns + ns_gen
    out_cnt_st = out_cnt_st + st_gen
    return out_cnt_ns, out_cnt_st, ns_gen, st_gen, output.astype("float32")


class NSbuilder(torch.nn.Module):
    """
    this module is the normalized stability builder.
    it converts the normalized stability of a bitstream into the desired value.
    """

    def __init__(
        self,
        bitwidth=8,
        mode="bipolar",
        normstability=0.5,
        threshold=0.05,
        value=None,
        rng_dim=1,
        rng="Sobol",
        rtype=torch.float,
        stype=torch.float,
    ):
        super(NSbuilder, self).__init__()

        self.bitwidth = bitwidth
        self.normstb = normstability
        if mode == "bipolar":
            self.val = (value + 1) / 2
            self.T = threshold / 2
        elif mode == "unipolar":
            self.val = value
            self.T = threshold
        self.mode = mode
        self.val_shape = self.val.size()
        self.val_dim = len(self.val_shape)

        self.stype = stype
        self.rtype = rtype

        self.L = torch.nn.Parameter(
            torch.tensor([2**self.bitwidth]).type(self.val.dtype), requires_grad=False
        )
        self.lp = torch.zeros_like(self.val)
        self.R = torch.ones_like(self.val)

        self.P_low = torch.zeros_like(self.val)
        self.P_up = torch.zeros_like(self.val)

        self.max_stable = torch.zeros_like(self.val)
        self.max_st_len = torch.zeros_like(self.val)
        self.new_st_len = torch.zeros_like(self.val)
        self.new_ns_len = torch.zeros_like(self.val)

        self.new_ns_val = torch.zeros_like(self.val)
        self.new_st_val = torch.zeros_like(self.val)
        self.new_ns_one = torch.zeros_like(self.val)
        self.new_st_one = torch.zeros_like(self.val)

        self.rng = RNG(bitwidth=bitwidth, dim=rng_dim, rng=rng, rtype=rtype)()

        self.ns_gen = torch.ones_like(self.val).type(torch.bool)
        self.st_gen = torch.zeros_like(self.val).type(torch.bool)

        self.out_cnt_ns = torch.zeros_like(self.val).type(torch.int32)
        self.out_cnt_st = torch.zeros_like(self.val).type(torch.int32)

        self.output = torch.zeros_like(self.val).type(stype)

        ## INIT:
        # Stage to calculate several essential params
        self.P_low = torch.max(self.val - self.T, torch.zeros_like(self.val))
        self.P_up = torch.min(torch.ones_like(self.val), self.val + self.T)
        upper = torch.min(torch.ceil(self.L * self.P_up), self.L)
        lower = torch.max(torch.floor(self.L * self.P_low), torch.zeros_like(self.L))

        seach_range = (self.T * 2 * self.L + 1).type(torch.int32)

        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab_parallel_numpy(
            lower.type(torch.int32).numpy(),
            upper.type(torch.int32).numpy(),
            self.L.type(torch.int32).numpy(),
            seach_range.numpy(),
        )

        self.max_stable = torch.from_numpy(max_stab_len)
        self.lp = torch.from_numpy(max_stab_l_p)
        self.R = torch.from_numpy(max_stab_R)

        self.max_st_len = self.L - (self.max_stable)
        self.new_st_len = torch.ceil(self.max_st_len * self.normstb)
        self.new_ns_len = self.L - self.new_st_len

        val_gt_half = (self.val > 0.5).type(torch.float)
        self.new_ns_one = val_gt_half * (self.P_up * (self.new_ns_len + 1)) + (
            1 - val_gt_half
        ) * torch.max(
            (self.P_low * (self.new_ns_len + 1) - 1), torch.zeros_like(self.L)
        )

        self.new_st_one = self.val * self.L - self.new_ns_one
        self.new_ns_val = self.new_ns_one / self.new_ns_len
        self.new_st_val = self.new_st_one / self.new_st_len

        self.src_st = SourceGen(
            self.new_st_val, self.bitwidth, "unipolar", self.rtype
        )()
        self.src_ns = SourceGen(
            self.new_ns_val, self.bitwidth, "unipolar", self.rtype
        )()
        self.bs_st = BSGen(self.src_st, self.rng)
        self.bs_ns = BSGen(self.src_ns, self.rng)

    def NSbuilder_forward(self):
        # parallel execution based on numpy speeds up by 90X
        ## Stage to generate output
        bs_st = self.bs_st(self.out_cnt_st).type(torch.int32).numpy()
        bs_ns = self.bs_ns(self.out_cnt_ns).type(torch.int32).numpy()
        out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out_parallel_numpy(
            self.new_ns_len.type(torch.int32).numpy(),
            self.out_cnt_ns.type(torch.int32).numpy(),
            self.out_cnt_st.type(torch.int32).numpy(),
            self.ns_gen.type(torch.int32).numpy(),
            self.st_gen.type(torch.int32).numpy(),
            bs_st,
            bs_ns,
            self.L.type(torch.int32).numpy(),
        )

        self.out_cnt_ns = torch.from_numpy(out_cnt_ns)
        self.out_cnt_st = torch.from_numpy(out_cnt_st)
        self.ns_gen = torch.from_numpy(ns_gen)
        self.st_gen = torch.from_numpy(st_gen)
        self.output = torch.from_numpy(output)

        return self.output.type(self.stype)

    def forward(self):
        return self.NSbuilder_forward()
