import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import random
import math
import time
from typing import Deque, Set, Dict, List, Tuple
import numpy as np

def make_ndarray_with_pad(
    x: List[List[int]],
    pad: int,
):
    max_len = max(map(len, x), default=0)
    padded_x = np.full((len(x), max_len), pad, dtype=int)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb
    return padded_x

def make_tensor_with_pad(
    x: List[List[int]],
    pad: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    padded_x = make_ndarray_with_pad(x, pad)
    tensor = torch.from_numpy(padded_x).to(device)
    return tensor

def merge(acc_attn_weights, acc_o, new_softmax_lse, new_o):
    bs = acc_o.shape[0]
    num_head = acc_o.shape[2]
    acc_attn_weights = acc_attn_weights.reshape(bs, 1, num_head, 1)
    new_softmax_lse = new_softmax_lse.reshape(bs, 1, num_head, 1)
    new_attn_weights = torch.exp(new_softmax_lse)
    updated_acc_attn_weights = acc_attn_weights + new_attn_weights
    w_acc = acc_attn_weights / updated_acc_attn_weights
    w_new = new_attn_weights / updated_acc_attn_weights
    acc_o = w_acc * acc_o + w_new * new_o
    return updated_acc_attn_weights, acc_o.type(torch.float16)

if __name__ == "__main__":
    num_blocks = 102400
    block_size = 32
    num_k_head = 4
    head_dim = 128
    kv_cache = torch.randn(
        (2, num_blocks, block_size, num_k_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    batch_size = 4
    num_head = 8
    query_len = 1
    query = torch.randn(
        (batch_size, query_len, num_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    max_key_len = 512
    max_num_blocks = max_key_len // block_size
    block_tables: List[List[int]] = []
    seq_lens: List[int] = []
    block_ids = range(0, num_blocks - 1)
    for _ in range(batch_size - 1):
        num_blocks = random.randint(0, max_num_blocks)
        block_tables.append(random.sample(block_ids, num_blocks))
        seq_lens.append(num_blocks * block_size)
    block_tables.append([])
    seq_lens.append(0)
    block_tables_tensor = make_tensor_with_pad(block_tables, device='cuda:0').type(torch.int32)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device='cuda:0')
    mask = (seq_lens_tensor == 0)
    print(mask)
    torch.cuda.synchronize()
    flash_output, softmax_lse, flash_attn_weights = flash_attn_with_kvcache(
        q=query,
        k_cache=key_cache,
        v_cache=value_cache,
        cache_seqlens=seq_lens_tensor,
        softmax_scale=head_dim**-0.5,
        rotary_interleaved=False,
        causal=True,
        alibi_slopes=None,
        block_table=block_tables_tensor,
        return_attn_weights=True,
    )
    softmax_lse = softmax_lse.reshape(batch_size, num_head)
    flash_attn_weights = flash_attn_weights.reshape(batch_size, num_head, -1)
    torch.cuda.synchronize()
    print(flash_output.shape)
    print(softmax_lse.shape)
    print(flash_attn_weights.shape)
    softmax_lse[mask] = 0
    print(softmax_lse)
    flash_output = flash_output.squeeze(1)
    print(flash_output.shape)
    block_tables = block_tables_tensor.tolist()
    print(block_tables)
    a = set([0, 1])
    b = set([2, 3])
    print(a.union(b))
    