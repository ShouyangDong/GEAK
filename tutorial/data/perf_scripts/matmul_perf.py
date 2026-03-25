# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os

# Add current directory to path for generated kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Import reference kernel from kernels directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
sys.path.insert(0, KERNELS_DIR)
from matmul import matmul_wrapper as matmul_wrapper_ref

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class matmul_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__("matmul", dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        """
        构建测试输入。
        对于 Matmul，性能通常随矩阵规模 (M, N, K) 的增大而提升，直至达到硬件峰值。
        我们这里扫描方阵规模。
        """
        self.input_tensors = []
        # 测试从 512 到 8192 的方阵
        for size in [512, 1024, 2048, 4096, 8192]:
            M, N, K = size, size, size
            a = torch.randn((M, K), dtype=self.dtype)
            b = torch.randn((K, N), dtype=self.dtype)

            # matmul 包装函数接受 (a, b, activation)
            args = (a, b, "")

            if self.is_backward:
                # 如果要测反向传播，需要准备 grad_output (do)
                # 注：你提供的代码仅包含 fwd，此处为反向框架预留
                do = torch.randn((M, N), dtype=self.dtype)
                self.input_tensors.append((*args, do))
            else:
                self.input_tensors.append(args)

    def to_mlu(self, input_tuple):
        """搬运至加速器并确保连续性（Kernel 要求）"""
        device = "mlu" if torch.mlu.is_available() else "cuda"
        # Matmul Kernel 显式要求输入连续
        tensors = [
            t.to(device).contiguous() if isinstance(t, torch.Tensor) else t
            for t in input_tuple
        ]

        if self.is_backward:
            tensors[0].requires_grad_()
            tensors[1].requires_grad_()
        return tuple(tensors)

    def call_op(self, input_tuple):
        """调用 Triton Matmul"""
        if self.is_backward:
            *args, do = input_tuple
            res = matmul(*args)
            return res.backward(do, retain_graph=True)
        else:
            return matmul(*input_tuple)

    def call_op_ref(self, input_tuple):
        """参考实现：PyTorch 原生调用 (通常调用 cuBLAS/cnBLAS)"""
        a, b, activation = input_tuple[:3]
        if self.is_backward:
            do = input_tuple[3]
            res = torch.matmul(a, b)
            if activation == "leaky_relu":
                res = torch.nn.functional.leaky_relu(res, negative_slope=0.01)
            return res.backward(do, retain_graph=True)
        else:
            res = torch.matmul(a, b)
            if activation == "leaky_relu":
                res = torch.nn.functional.leaky_relu(res, negative_slope=0.01)
            return res

    def get_tflops(self, input_tuple, runtime):
        """
        计算 TFLOPS。
        Matmul 计算量公式：2 * M * N * K (乘加算作两次运算)
        """
        a, b = input_tuple[0], input_tuple[1]
        M, K = a.shape
        _, N = b.shape

        flops = 2 * M * N * K
        if self.is_backward:
            # 矩阵乘法反向传播包含两个矩阵乘法 (dA = dC @ B^T, dB = A^T @ dC)
            flops *= 2

        return flops / (runtime / 1000) / 1e12

    def get_gbps(self, input_tuple, runtime):
        """
        计算 GB/s。
        对于计算密集型算子，此数值通常远低于显存带宽峰值。
        """
        a, b = input_tuple[0], input_tuple[1]
        M, K = a.shape
        _, N = b.shape

        # 读 A, B + 写 C
        total_bytes = (M * K + K * N + M * N) * a.element_size()
        return total_bytes / (runtime / 1000) / 1e9


if __name__ == "__main__":
    # 执行性能评测
    perf = matmul_performance_metrics(is_backward=False)
    perf.get_input_tensors()
    # Matmul 运行较快，建议 rep 设置大一些以获得稳定值
    perf.get_do_bench_config(warmup=50, rep=200)
    perf.run_benchmark()
