# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os

# Add current directory to path for generated kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Import reference kernel from kernels directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
sys.path.insert(0, KERNELS_DIR)
from layernorm import layernorm_wrapper as layernorm_wrapper_ref

from performance_utils import Performance_Metrics, do_bench_config


import torch
import triton
import triton.language as tl


class layernorm_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__("layernorm", dtype=dtype, is_backward=is_backward, **kwargs)
        self.eps = 1e-5

    def get_input_tensors(self):
        """
        构建测试输入。
        LayerNorm 的性能主要受两个维度影响：
        M (Batch * SeqLen): 决定了并行执行的 Row 数量。
        N (Hidden Dimension): 决定了每个 Block 的计算负载和归约压力。
        """
        self.input_tensors = []
        # 固定 M，扫描 N (特征维度) 以观察两阶段归约的效率
        M = 4096
        for N in [512, 1024, 2048, 4096, 8192, 16384]:
            x = torch.randn((M, N), dtype=self.dtype)
            weight = torch.ones(N, dtype=self.dtype)
            bias = torch.zeros(N, dtype=self.dtype)

            args = (x, (N,), weight, bias, self.eps)

            if self.is_backward:
                # 预演一遍 forward 以获取反向传播所需的 ctx
                dy = torch.randn_like(x)
                self.input_tensors.append((*args, dy))
            else:
                self.input_tensors.append(args)

    def to_mlu(self, input_tuple):
        """将张量搬运至加速器并设置梯度追踪"""
        device = "mlu"
        tensors = [
            t.to(device) if isinstance(t, torch.Tensor) else t for t in input_tuple
        ]

        if self.is_backward:
            tensors[0].requires_grad_()  # x
            tensors[2].requires_grad_()  # weight
            tensors[3].requires_grad_()  # bias
        return tuple(tensors)

    def call_op(self, input_tuple):
        """调用 Triton 实现的 LayerNorm"""
        if self.is_backward:
            *args, dy = input_tuple
            y = layernorm_wrapper_ref.apply(*args)
            return y.backward(dy, retain_graph=True)
        else:
            return layernorm_wrapper_ref.apply(*input_tuple)

    def call_op_ref(self, input_tuple):
        """参考实现：使用 PyTorch 原生的 F.layer_norm"""
        if self.is_backward:
            *args, dy = input_tuple
            x, normalized_shape, weight, bias, eps = args
            y = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
            return y.backward(dy, retain_graph=True)
        else:
            return torch.nn.functional.layer_norm(*input_tuple)

    def get_gbps(self, input_tuple, runtime):
        """
        计算显存带宽利用率 (GB/s)
        """
        x = input_tuple[0]
        M, N = x.shape
        element_size = x.element_size()

        if not self.is_backward:
            total_bytes = (M * N * 2 * element_size) + (M * 2 * 4)
        else:
            total_bytes = (
                (M * N * 4 * element_size) + (M * 2 * 4) + (N * 2 * element_size)
            )

        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        """
        LayerNorm 的计算量相对较小，主要是元素级操作。
        计算量公式约为：10 * M * N (包含求和、平方、归一化等)
        """
        x = input_tuple[0]
        M, N = x.shape
        flops = 10 * M * N
        if self.is_backward:
            flops *= 2  # 反向计算量通常为前向的 2-3 倍
        return flops / (runtime / 1000) / 1e12


if __name__ == "__main__":
    bwd_perf = layernorm_performance_metrics(is_backward=True)
    bwd_perf.get_input_tensors()
    bwd_perf.get_do_bench_config(warmup=50, rep=200)
    bwd_perf.run_benchmark()
