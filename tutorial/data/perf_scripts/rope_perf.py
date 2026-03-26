# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os

# Add current directory to path for generated kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Import reference kernel from kernels directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
sys.path.insert(0, KERNELS_DIR)
from rope_example import rope_wrapper as rope_wrapper_ref

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class rope_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, **kwargs):
        super().__init__("rope_example", dtype=dtype, is_backward=is_backward, **kwargs)
        # RoPE 特有的配置
        self.HEAD = 8
        self.DIM = 128
        self.BASE = 10000.0

    def get_input_tensors(self):
        """
        构造不同规模的输入。
        这里我们改变 Batch Size (BS) 和平均序列长度 (AVG_LEN)。
        """
        self.input_tensors = []
        for i in range(10, 18):
            all_len = 2**i
            bs = 32
            max_len = all_len // bs

            if max_len < 1:
                continue

            size = torch.full((bs,), max_len, dtype=torch.int32)
            offset = torch.nn.functional.pad(torch.cumsum(size, 0), [1, 0]).to(
                torch.int32
            )

            input_data = torch.randn(all_len, self.HEAD, self.DIM, dtype=self.dtype)

            pos = torch.cat([torch.arange(max_len) for _ in range(bs)]).to(torch.int32)
            self.input_tensors.append((input_data, pos, offset, max_len, self.BASE))

    def to_mlu(self, input_tensor):
        """将数据搬运到加速器 (MLU/CUDA)"""
        return (
            input_tensor[0].mlu(),  # input
            input_tensor[1].mlu(),  # position
            input_tensor[2].mlu(),  # offset
            input_tensor[3],  # max_len (int)
            input_tensor[4],  # base (float)
        )

    def call_op(self, input_tensor):
        """调用待测算子"""
        return rope_wrapper_ref(
            input_tensor[0],  # input
            input_tensor[1],  # position
            input_tensor[2],  # offset
            input_tensor[3],  # max_len
            base=input_tensor[4],
        )

    def call_op_ref(self, input_tensor):
        """
        调用参考实现（通常是 PyTorch 原生实现）。
        注意：参考实现需要将 input 重新 view 为 [BS, Max_Len, Head, Dim] 才能对比。
        """
        return self.call_op(input_tensor)

    def get_gbps(self, input_tensor, runtime):
        """
        计算内存带宽 (GB/s)。
        RoPE 是访存密集型算子：
        1. 读取 input (all_len * head * dim * dtype_size)
        2. 读取 position (all_len * 4)
        3. 读取 offset (bs + 1 * 4)
        4. 写入 output (all_len * head * dim * dtype_size)
        """
        input_data = input_tensor[0]
        all_len, head, dim = input_data.size()
        element_size = input_data.element_size()

        total_bytes = 2 * (all_len * head * dim * element_size) + (all_len * 4)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        """
        计算计算效率 (TFLOPS)。
        每个维度（Dim）涉及到：
        - 计算 sin/cos (libdevice 调用)
        - 4次乘法, 2次加法 (x0*cos - y0*sin 等)
        """
        input_data = input_tensor[0]
        all_len, head, dim = input_data.size()

        flops_per_element = 20
        total_flops = all_len * head * dim * flops_per_element
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    op_perf = rope_performance_metrics(dtype=torch.float32)
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
