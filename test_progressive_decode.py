import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import random
import math
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
    attention = torch.softmax(attention_scores, dim=-1)
    softmax_lse = torch.log(torch.sum(torch.exp(attention_scores), dim=-1))
    # print(softmax_lse.shape)
    attention = attention.type(value.dtype)
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
    return output, attention_scores, softmax_lse

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
    num_blocks = 10240
    block_size = 32
    num_k_head = 32
    head_dim = 128
    kv_cache = torch.randn(
        (2, num_blocks, block_size, num_k_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    batch_size = 4
    num_head = 32
    query_len = 1
    local_window = 4096
    key_len = 512
    query = torch.randn(
        (batch_size, query_len, num_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    
    key_lens_tensor = torch.ones((batch_size, ), dtype=torch.int32, device='cuda:0') * key_len
    key_lens_tensor[0] = 512
    key_lens_tensor[1] = 512
    key_lens_tensor[2] = 512
    key_lens_tensor[3] = 512
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
        flash_output0, flash_attn_weights_sum, flash_attn_weights_min = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_seqlens=key_lens_tensor,
            softmax_scale=head_dim**-0.5,
            rotary_interleaved=False,
            causal=True,
            alibi_slopes=None,
            block_table=block_tables,
            return_attn_weights=True
        )
        torch.cuda.synchronize()
        if i >= warmup:
            total_time += time.time()
    print("flash attention time cost without attn scores: ", total_time / (num_pass - warmup) * 1000)
    
    exit(0)
    
    org_latency = total_time / (num_pass - warmup)
    flash_attn_weights = flash_attn_weights.reshape(batch_size, num_head, key_len // block_size)
    softmax_lse = softmax_lse.reshape(batch_size, num_head)
    print(flash_attn_weights.shape)
    print(softmax_lse.shape)
    flash_attn_weights_min = flash_attn_weights_min.reshape(batch_size, num_head)
    for i in range(batch_size):
        key_len = key_lens_tensor[i]
        num_pages = key_len // block_size
        for head_idx in range(num_head):
            print("sum: ", torch.exp(softmax_lse[i][head_idx]), torch.sum(flash_attn_weights[i][head_idx][:num_pages]))
            print("min: ", flash_attn_weights_min[i][head_idx], torch.min(flash_attn_weights[i][head_idx][:num_pages], -1)[0])
    
    exit(0)
    
    flash_attn_weights = torch.sum(flash_attn_weights, dim=2).flatten()
    flash_attn_weights = flash_attn_weights.reshape(softmax_lse.shape[0], num_k_head, -1)
    print(flash_attn_weights.shape)
    for i in range(8):
        per_head_attn_weights = flash_attn_weights[0][i]
        per_head_attn_weights = per_head_attn_weights.flatten().tolist()
        per_head_attn_weights.sort(reverse=True)
        print(per_head_attn_weights)
    # softmax_scale = head_dim**-0.5
    # flash_attn_scores = flash_attn_scores.type(torch.float32)
    # flash_attn_scores = torch.sum(torch.exp(flash_attn_scores * softmax_scale), dim=-1).flatten()
    flash_attn_weights = torch.sum(flash_attn_weights, dim=-1).flatten()
    softmax_lse = torch.exp(softmax_lse).flatten()
    diff = torch.sum(torch.abs(softmax_lse - flash_attn_weights))
    print("output diff: ", diff)
    # diff = torch.sum(torch.abs(softmax_lse - flash_attn_scores))
    # # print(softmax_lse.shape, flash_attn_scores.shape, flash_attn_weights.shape)
    # for i in range(len(softmax_lse)):
    #     if i % 17 == 0:
    #         print(softmax_lse[i], flash_attn_weights[i])
    # print("output diff: ", diff)
    
    # softmax_scale = head_dim**-0.5
    # flash_attn_scores = torch.log(torch.sum(torch.exp(flash_attn_scores.type(torch.float32) * softmax_scale), dim=-1))
    # flash_attn_scores = flash_attn_scores.transpose(1, 2).reshape(batch_size, num_head)
    # flash_attn_scores = flash_attn_scores.flatten()
    # flash_attn_weights = flash_attn_weights.reshape(batch_size, num_head)
    # flash_attn_weights = flash_attn_weights.flatten()
    # diff = 0
    # for idx in range(flash_attn_scores.stride(0)):
    #     diff += torch.abs(flash_attn_scores[idx] - flash_attn_weights[idx])
    # print("output diff: ", diff)
    