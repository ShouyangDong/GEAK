# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os

# Add current directory to path for generated kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import reference kernel from kernels directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
sys.path.insert(0, KERNELS_DIR)
from flash_attention import flash_attention_wrapper as flash_attention_wrapper_ref

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class flash_attention_performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float16, is_backward=False, **kwargs):
        super().__init__(
            "flash_attention", dtype=dtype, is_backward=is_backward, **kwargs
        )
        self.head_q = 32
        self.head_kv = 32  # 可以修改为 GQA/MQA 模式进行测试
        self.dim = 64
        self.scale = 1.0 / (self.dim**0.5)

    def get_input_tensors(self):
        """
        构建测试输入：改变序列长度 Sequence Length
        """
        self.input_tensors = []
        # 测试序列长度从 1024 到 8192
        for seq_len in [1024, 2048, 4096, 8192]:
            batch = 1
            q = torch.randn((batch * seq_len, self.head_q, self.dim), dtype=self.dtype)
            k = torch.randn((batch * seq_len, self.head_kv, self.dim), dtype=self.dtype)
            v = torch.randn((batch * seq_len, self.head_kv, self.dim), dtype=self.dtype)

            # 构造 Flash Attention 专用的辅助张量
            cu_seqlens = torch.arange(
                0, (batch + 1) * seq_len, seq_len, dtype=torch.int32
            )
            # 模拟简单的 mask 参数，attn_arg 这里填 0 (全序列)
            q_attn_arg = torch.zeros(batch * seq_len, dtype=torch.int32)
            k_attn_arg = torch.zeros(batch * seq_len, dtype=torch.int32)

            args = (
                q,
                k,
                v,
                q_attn_arg,
                k_attn_arg,
                cu_seqlens,
                cu_seqlens,
                seq_len,
                seq_len,
                self.scale,
                1,
                False,
            )

            if self.is_backward:
                do = torch.randn(
                    (batch * seq_len, self.head_q, self.dim), dtype=self.dtype
                )
                self.input_tensors.append((*args, do))
            else:
                self.input_tensors.append(args)

    def to_mlu(self, input_tuple):
        """搬运至加速器"""
        device = "mlu"
        tensors = [
            t.to(device) if isinstance(t, torch.Tensor) else t for t in input_tuple
        ]
        if self.is_backward:
            # 激活 QKV 的梯度
            for t in tensors[:3]:
                t.requires_grad_()
        return tuple(tensors)

    def call_op(self, input_tuple):
        """调用 FlashAttentionFunc"""
        if self.is_backward:
            *args, do = input_tuple
            o = flash_attention_wrapper_ref.apply(*args)
            return torch.autograd.backward(o, do, retain_graph=True)
        else:
            return flash_attention_wrapper_ref.apply(*input_tuple)

    def call_op_ref(self, input_tuple):
        """
        参考实现：标准的 Causal Attention 逻辑。
        注意：原生实现对于长序列会 OOM，测试时请注意 seq_len 上限。
        """
        q, k, v, _, _, cu_q, cu_k, max_q, max_k, scale, mask_type, sparse = input_tuple[
            :12
        ]
        # 这里仅实现简化的单 batch 逻辑作为参考
        q = q.transpose(0, 1)  # [H, L, D]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        # 简单模拟 causal mask
        mask = torch.triu(torch.ones(max_q, max_k, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        p = torch.softmax(attn, dim=-1)
        o = torch.matmul(p, v)

        if self.is_backward:
            do = input_tuple[12]
            return torch.autograd.backward(o, do.transpose(0, 1), retain_graph=True)
        return o

    def get_tflops(self, input_tuple, runtime):
        """
        计算 TFLOPS。
        Attention 的计算量公式约为：2 * L^2 * H * D (包含 QK^2 和 PV 两次大矩阵乘)
        """
        q = input_tuple[0]
        L = input_tuple[7]  # max_seqlen_q
        H = self.head_q
        D = self.dim

        # 前向计算量：QK^T (2L^2HD) + Softmax (忽略) + PV (2L^2HD)
        flops = 4 * L**2 * H * D
        if self.is_backward:
            # 反向计算量约为前向的 2.5 到 3 倍
            flops *= 2.5

        return flops / (runtime / 1000) / 1e12

    def get_gbps(self, input_tuple, runtime):
        """
        对于 Flash Attention，由于中间矩阵 P 不写回显存，GB/s 通常不是瓶颈。
        主要考察 Q, K, V 的 IO。
        """
        q, k, v = input_tuple[0], input_tuple[1], input_tuple[2]
        # 读 Q, K, V + 写 O
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        return total_bytes / (runtime / 1000) / 1e9


if __name__ == "__main__":
    # 测试前向
    fwd_perf = flash_attention_performance_metrics(is_backward=False)
    fwd_perf.get_input_tensors()
    fwd_perf.get_do_bench_config(warmup=20, rep=50)
    fwd_perf.run_benchmark()
