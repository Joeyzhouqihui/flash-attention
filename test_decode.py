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

def checkAttnScores(flash_attn_scores, torch_attn_scores, batch_size, num_head, num_head_k, query_len, key_len, local_window):
    assert num_head % num_head_k == 0
    group_size = int(num_head / num_head_k)
    assert flash_attn_scores.shape[1] == num_head_k
    assert flash_attn_scores.shape[2] == group_size
    assert torch_attn_scores.shape[1] == num_head
    assert torch_attn_scores.shape[2] == 1
    diff = 0.0
    for batch in range(batch_size):
        for row in range(num_head_k):
            for col in range(group_size):
                head = row * group_size + col
                flash_slice = flash_attn_scores[batch][row][col][-local_window:]
                torch_slice = torch_attn_scores[batch][head][0][-local_window:]
                diff += torch.sum(flash_slice - torch_slice)
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
    batch_size = 32
    num_head = 8
    query_len = 1
    local_window = 512
    key_len = 128
    query = torch.randn(
        (batch_size, query_len, num_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    
    key_lens_tensor = torch.ones((batch_size, ), dtype=torch.int32, device='cuda:0') * key_len
    
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
        flash_output0 = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_seqlens=key_lens_tensor,
            softmax_scale=head_dim**-0.5,
            rotary_interleaved=False,
            causal=True,
            alibi_slopes=None,
            block_table=block_tables,
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
        flash_output1, softmax_lse, flash_attn_weights = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_seqlens=key_lens_tensor,
            softmax_scale=head_dim**-0.5,
            causal=True,
            rotary_interleaved=False,
            alibi_slopes=None,
            block_table=block_tables,
            num_local_tokens=0,
            return_attn_scores=False,
            reduce_attn_scores=False,
            return_attn_weights=True
        )
        torch.cuda.synchronize()
        if i >= warmup:
            total_time += time.time()
    print("flash attention time cost with attn scores: ", total_time / (num_pass - warmup) * 1000)
    # print("flash attn scores shape: ", flash_attn_scores.shape)
    print("flash attn weight shape: ", flash_attn_weights.shape)
    exit(0)
    # total_time = 0
    # for i in range(num_pass):
    #     torch.cuda.synchronize()
    #     if i >= warmup:
    #         total_time -= time.time()
    #     flash_output2, flash_attn_weights2, flash_attn_scores2 = flash_attn_with_kvcache(
    #         q=query,
    #         k_cache=key_cache,
    #         v_cache=value_cache,
    #         cache_seqlens=key_lens_tensor,
    #         softmax_scale=head_dim**-0.5,
    #         causal=True,
    #         rotary_interleaved=False,
    #         alibi_slopes=None,
    #         block_table=block_tables,
    #         num_local_tokens=local_window,
    #         return_attn_scores=True,
    #         reduce_attn_scores=True
    #     )
        
    #     torch.cuda.synchronize()
    #     if i >= warmup:
    #         total_time += time.time()
    # print("flash attention time cost with attn scores reduce: ", total_time / (num_pass - warmup) * 1000)
    # print("flash attn scores with attn scores reduce shape: ", flash_attn_scores2.shape)
    # print("flash attn weight shape: ", flash_attn_weights2.shape)
    
    query = query.reshape((batch_size, -1, num_head, head_dim))
    key = key_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    value = value_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    torch_output, torch_attn_scores, torch_attn_weights = attention_pytorch(query, key, value, local_window)
    print("torch attn scores shape: ", torch_attn_scores.shape)
    print("torch attn weights shape: ", torch_attn_weights.shape)

    # assert flash_output0.shape == torch_output.shape
    # assert flash_output1.shape == torch_output.shape
    # assert flash_output2.shape == torch_output.shape
    # assert flash_attn_weights1.shape == torch_attn_weights.shape
    # assert flash_attn_weights2.shape == torch_attn_weights.shape
    
    # flash_attn_weights1 = flash_attn_weights1.flatten()
    # flash_attn_weights2 = flash_attn_weights2.flatten()
    torch_attn_weights = torch_attn_weights.flatten()
    # diff1 = torch.sum(torch.abs(flash_attn_weights - flash_attn_weights2))
    diff2 = torch.sum(torch.abs(flash_attn_weights - torch_attn_weights))
    # for i in range(len(flash_attn_weights1)):
    #     if i % 7 == 0:
    #         print(flash_attn_weights1[i], torch_attn_weights[i])
    # assert diff1 == 0
    
    # flash_output0 = flash_output0.flatten()
    # flash_output1 = flash_output1.flatten()
    flash_output2 = flash_output2.flatten()
    torch_output = torch_output.flatten()
    # diff = 0
    # for idx in range(flash_output0.stride(0)):
    #     diff += torch.abs(flash_output0[idx] - torch_output[idx])
    # print("output diff: ", diff)
    # diff = 0
    # for idx in range(flash_output1.stride(0)):
    #     diff += torch.abs(flash_output1[idx] - torch_output[idx])
    # print("output diff: ", diff)
    diff = 0
    for idx in range(flash_output2.stride(0)):
        diff += torch.abs(flash_output2[idx] - torch_output[idx])
    print("output diff: ", diff)
    
    # print(torch_attn_scores[0][0][0][-10:])
    # print(flash_attn_scores[0][0][0][-10:])
    # print(flash_attn_scores2[0][0][0][-10:])
    checkAttnScores(flash_attn_scores2, torch_attn_scores, batch_size, num_head, num_k_head, query_len, key_len, local_window)

    