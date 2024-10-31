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
    attention_scores = torch.matmul(query, key.transpose(-2, -1)).type(torch.float32)
    causal_mask = torch.triu(torch.full((seq_len_key, seq_len_key), True, device=attention_scores.device), 1)
    causal_mask = causal_mask[-seq_len_query:, :]
    attention_scores.masked_fill_(causal_mask, 0)
    return output, attention_scores.type(torch.float16), attention_scores

def checkAttnScores(flash_attn_scores, torch_attn_scores, batch_size, num_head, query_len, key_len, local_window):
    assert flash_attn_scores.shape == torch_attn_scores.shape
    assert query_len <= key_len
    assert local_window > 0
    diff = 0.0
    for batch in range(batch_size):
        for head in range(num_head):
            for row in range(query_len):
                last_token = key_len - query_len + row
                first_token = max(0, last_token - local_window + 1)
                flash_slice = flash_attn_scores[batch][head][row][first_token : last_token + 1]
                torch_slice = torch_attn_scores[batch][head][row][first_token : last_token + 1]
                diff += torch.abs(torch.sum(flash_slice - torch_slice))
    assert diff == 0
    print("attention scores diff: ", diff)
    
def checkAttnScoresReduce(flash_attn_scores, torch_attn_scores, batch_size, num_head, query_len, key_len, local_window):
    assert query_len <= key_len
    assert local_window > 0
    chunk_size = 64
    assert flash_attn_scores.shape[2] == math.ceil(torch_attn_scores.shape[2] / 64)
    diff = 0.0
    for batch in range(batch_size):
        for head in range(num_head):
            for chunk_id in range(flash_attn_scores.shape[2]):
                offset = key_len - query_len
                first_token = key_len - query_len + chunk_id * chunk_size
                last_token = min(key_len - 1, first_token + chunk_size - 1)
                left = max(0, first_token - local_window + 1)
                flash_slice = flash_attn_scores[batch][head][chunk_id][left : last_token + 1]
                torch_slice = torch.zeros(key_len).type(torch.float32).to(0)
                for token in range(first_token, last_token + 1):
                    left = max(0, token - local_window + 1)
                    torch_slice[left : token + 1] += torch_attn_scores[batch][head][token - offset][left : token + 1]
                left = max(0, first_token - local_window + 1)
                torch_slice = torch_slice[left : last_token + 1].type(torch.float16)
                if chunk_id == 0 and batch == 0 and head == 7:
                    print(flash_slice[0:10])
                    print(torch_slice[0:10])
                    print(flash_slice[-10:])
                    print(torch_slice[-10:])
                    return
                diff += torch.abs(torch.sum(flash_slice - torch_slice))
    print("attention scores diff: ", diff)
    assert diff == 0

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
    batch_size = 2
    num_head = 64
    query_len = 512
    local_window = 512
    key_len = 2048
    query = torch.randn(
        (batch_size * query_len, num_head, head_dim),                    
        dtype=torch.float16, 
        device='cuda:0'
    )
    
    query_lens_tensor = torch.ones((batch_size), dtype=torch.int32, device='cuda:0') * query_len
    cu_seqlens_q = torch.zeros((batch_size + 1), dtype=torch.int32, device='cuda:0')
    torch.cumsum(query_lens_tensor, dim=0, dtype=cu_seqlens_q.dtype, out=cu_seqlens_q[1:])
    
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
        flash_output0 = flash_attn_varlen_func(
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
        flash_output1, flash_attn_scores1 = flash_attn_varlen_func(
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
    print("flash attn scores shape: ", flash_attn_scores1.shape)
    
    total_time = 0
    for i in range(num_pass):
        torch.cuda.synchronize()
        if i >= warmup:
            total_time -= time.time()
        flash_output2, flash_attn_scores2 = flash_attn_varlen_func(
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
            return_attn_scores=True,
            reduce_attn_scores=True
        )
        torch.cuda.synchronize()
        if i >= warmup:
            total_time += time.time()
    print("flash attention time cost with attn scores reduce: ", total_time / (num_pass - warmup) * 1000)
    print("flash attn scores with attn scores reduce shape: ", flash_attn_scores2.shape)
    print("flash output shape: ", flash_output2.shape)
    
    query = query.reshape((batch_size, -1, num_head, head_dim))
    key = key_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    value = value_cache[block_tables].reshape((batch_size, -1, num_k_head, head_dim))
    torch_output, torch_attn_scores_16, torch_attn_scores_32 = attention_pytorch(query, key, value, local_window)
    print("torch attn scores shape: ", torch_attn_scores_16.shape)
    
    flash_output0 = flash_output0.flatten()
    flash_output1 = flash_output1.flatten()
    flash_output2 = flash_output2.flatten()
    torch_output = torch_output.flatten()
    
    diff = 0
    for idx in range(flash_output0.stride(0)):
        diff += torch.abs(flash_output0[idx] - torch_output[idx])
    print("output diff: ", diff)
    diff = 0
    for idx in range(flash_output1.stride(0)):
        diff += torch.abs(flash_output1[idx] - torch_output[idx])
    print("output diff: ", diff)
    diff = 0
    for idx in range(flash_output2.stride(0)):
        diff += torch.abs(flash_output2[idx] - torch_output[idx])
    print("output diff: ", diff)
    
    checkAttnScores(flash_attn_scores1, torch_attn_scores_16, batch_size, num_head, query_len, key_len, local_window)
    checkAttnScoresReduce(flash_attn_scores2, torch_attn_scores_32, batch_size, num_head, query_len, key_len, local_window)
    # left = max(0, key_len-query_len-local_window+1)
    # print(flash_attn_scores2[0][0][0][left:left+10])
    # print(torch_attn_scores[0][0][0][left:left+10])
    # print(flash_attn_scores2[0][0][0][-10:])
    # print(torch_attn_scores[0][0][0][-10:])
    # checkAttnScoresReduce(flash_attn_scores2, torch_attn_scores_reduce, batch_size, num_head, query_len, key_len, local_window)
    
    # print(flash_attn_scores.shape, torch_attn_scores.shape)
    # flash_attn_scores = torch.sum(flash_attn_scores, dim=2).type(torch.float16)
    # print(flash_attn_scores.shape, torch_attn_scores_reduce.shape)
    # print(torch.sum(torch_attn_scores, dim=2))
    # checkAttnScores(flash_attn_scores, torch_attn_scores, batch_size, num_head, query_len, key_len, local_window)
    # print(flash_attn_scores.shape)
    # print(flash_attn_scores[0][0][0:128])
    # print(flash_attn_scores[0][0][128:256])
    # print(flash_attn_scores[0][0][256:512])
    # print(flash_attn_scores[0][0][512:512])
    # checkAttnScoresReduce(flash_attn_scores, torch_attn_scores_reduce, batch_size, num_head, query_len, key_len, local_window)
    # len_q = 3
    # len_k = 8
    # ws = 5
    # causal_mask = torch.logical_not(torch.logical_xor(
    #     torch.triu(torch.full((len_k, len_k), True), 1),
    #     torch.triu(torch.full((len_k, len_k), True), -ws+1)
    # ))
    # causal_mask = causal_mask[-len_q:,:]
    # print(causal_mask)