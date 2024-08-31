import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import random
import math
from einops import rearrange, repeat
import time

def attention_pytorch(query, key, value, local_window):
    batch_size, seq_len_query, n_heads, head_dim = query.shape
    _, seq_len_key, n_kheads, _ = key.shape
    query = query.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)  
    key = key.view(batch_size, -1, n_kheads, head_dim).repeat_interleave(n_heads//n_kheads, dim=2).transpose(1, 2)
    value = value.view(batch_size, -1, n_kheads, head_dim).repeat_interleave(n_heads//n_kheads, dim=2).transpose(1, 2)
    softmax_scale = head_dim**-0.5
    attention_scores = torch.matmul(query, key.transpose(-2, -1)).type(torch.float32)
    attention_scores = attention_scores * softmax_scale
    causal_mask = torch.triu(torch.full((seq_len_key, seq_len_key), True, device=attention_scores.device), 1)
    causal_mask = causal_mask[-seq_len_query:, :]
    attention_scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(attention_scores, dim=-1).type(value.dtype)
    output = torch.matmul(attention, value).transpose(1, 2)
    # window_mask = torch.logical_not(torch.logical_xor(
    #     torch.triu(torch.full((seq_len_key, seq_len_key), True, device=attention_scores.device), 1),
    #     torch.triu(torch.full((seq_len_key, seq_len_key), True, device=attention_scores.device), -local_window + 1)
    # ))
    # window_mask = window_mask[-seq_len_query:, :]
    # attention_scores.masked_fill_(window_mask, 0)
    # local_window_attention_scores = torch.sum(attention_scores, dim = -1)
    # print(local_window_attention_scores.shape)
    # print(attention_scores.shape)
    # # check results
    # diff = 0
    # for batch in range(batch_size):
    #     for head in range(num_head):
    #         for row in range(seq_len_query):
    #             all_score = []
    #             for col in range(seq_len_key):
    #                 all_score.append(attention_scores[batch][head][row][col])
    #             all_score = torch.tensor(all_score, dtype=local_window_attention_scores.dtype, device=local_window_attention_scores.device)
    #             diff += torch.abs(torch.sum(all_score) - local_window_attention_scores[batch][head][row])
    # assert diff == 0
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    causal_mask = torch.triu(torch.full((seq_len_key, seq_len_key), True, device=attention_scores.device), 1)
    causal_mask = causal_mask[-seq_len_query:, :]
    attention_scores.masked_fill_(causal_mask, float('-inf'))
    return output, attention_scores

def checkAttnScores(flash_attn_scores, torch_attn_scores, batch_size, num_head, query_len, key_len, local_window):
    diff = 0.0
    for batch in range(batch_size):
        for head in range(num_head):
            for row in range(query_len):
                end = min(key_len - query_len + row, row + local_window - 1)
                diff += torch.sum(flash_attn_scores[batch][head][row][0:end] - torch_attn_scores[batch][head][row][0:end])
                # for col in range(end):
                #     diff += torch.abs(flash_attn_scores[batch][head][row][col]- 
                #                       torch_attn_scores[batch][head][row][col])
                # if diff != 0:
                #     print(batch, head, row)
                #     assert False
    assert diff == 0
    print("attention scores diff: ", diff)

if __name__ == "__main__":
    num_blocks = 102400
    block_size = 16
    num_k_head = 8
    head_dim = 128
    kv_cache = torch.randn(
        (2, num_blocks, block_size, num_k_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    batch_size = 4
    num_head = 64
    query_len = 512
    local_window = 1024
    query = torch.randn(
        (batch_size * query_len, num_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    
    query_lens_tensor = torch.ones((batch_size), dtype=torch.int32, device='cuda:0') * query_len
    cu_seqlens_q = torch.zeros((batch_size + 1), dtype=torch.int32, device='cuda:0')
    torch.cumsum(query_lens_tensor, dim=0, dtype=cu_seqlens_q.dtype, out=cu_seqlens_q[1:])
    
    key_len = 8192
    key_lens_tensor = torch.ones((batch_size), dtype=torch.int32, device='cuda:0') * key_len
    cu_seqlens_k = torch.zeros((batch_size + 1), dtype=torch.int32, device='cuda:0')
    torch.cumsum(key_lens_tensor, dim=0, dtype=cu_seqlens_k.dtype, out=cu_seqlens_k[1:])
    
    block_ids = range(0, num_blocks - 1)
    block_tables = random.sample(block_ids, key_len // block_size * batch_size)
    block_tables = torch.tensor(block_tables,
        dtype=torch.int32, device='cuda:0'
    ).reshape((batch_size, key_len // block_size))
    
    torch.cuda.synchronize()
    
    num_pass = 16
    warmup = 8
    total_time = 0
    for i in range(num_pass):
        torch.cuda.synchronize()
        if i >= warmup:
            total_time -= time.time()
        flash_output = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=int(query_len),
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=int(key_len),
            softmax_scale=head_dim**-0.5,
            causal=True,
            alibi_slopes=None,
            block_table=block_tables,
            num_local_tokens=int(40),
            return_attn_scores=False
        )
        torch.cuda.synchronize()
        if i >= warmup:
            total_time += time.time()
    print("flash attention time cost without attn scores: ", total_time / (num_pass - warmup) * 1000)
    org_latency = total_time / (num_pass - warmup)
    
    total_time = 0
    for i in range(num_pass):
        torch.cuda.synchronize()
        if i >= warmup:
            total_time -= time.time()
        flash_output, flash_attn_scores = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=int(query_len),
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=int(key_len),
            softmax_scale=head_dim**-0.5,
            causal=True,
            alibi_slopes=None,
            block_table=block_tables,
            num_local_tokens=local_window,
            return_attn_scores=True
        )
        torch.cuda.synchronize()
        if i >= warmup:
            total_time += time.time()
    print("flash attention time cost with attn scores: ", total_time / (num_pass - warmup) * 1000)
    cur_latency = total_time / (num_pass - warmup)
    
    print("slow down ratio: {0}", cur_latency / org_latency)
    
    query = query.reshape((batch_size, -1, num_head, head_dim))
    key = key_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    value = value_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    torch_output, torch_attn_scores = attention_pytorch(query, key, value, local_window)
    
    flash_output = flash_output.flatten()
    torch_output = torch_output.flatten()
    diff = 0
    for idx in range(flash_output.stride(0)):
        diff += torch.abs(flash_output[idx] - torch_output[idx])
    print("output diff: ", diff)
    print(flash_attn_scores.shape, torch_attn_scores.shape)
    # print(flash_attn_scores[0][0])
    # print(torch_attn_scores[0][0])
    checkAttnScores(flash_attn_scores, torch_attn_scores, batch_size, num_head, query_len, key_len, local_window)
    # len_q = 3
    # len_k = 8
    # ws = 5
    # causal_mask = torch.logical_not(torch.logical_xor(
    #     torch.triu(torch.full((len_k, len_k), True), 1),
    #     torch.triu(torch.full((len_k, len_k), True), -ws+1)
    # ))
    # causal_mask = causal_mask[-len_q:,:]
    # print(causal_mask)