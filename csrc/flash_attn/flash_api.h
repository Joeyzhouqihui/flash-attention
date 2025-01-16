#pragma once

#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>
#include <semaphore.h>
#include <atomic>

#include "flash.h"
#include "static_switch.h"

void
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<at::Tensor> &seqlens_k_, // batch_size
                c10::optional<at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                c10::optional<at::Tensor> &out_es_sum_,             // batch_size x seqlen_q x num_heads x head_size
                c10::optional<at::Tensor> &out_es_min_,             // batch_size x seqlen_q x num_heads x head_size
                c10::optional<at::Tensor> &out_es_buffer_,          // batch_size x seqlen_q x num_heads x head_size
                float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                int num_local_tokens,
                bool return_attn_weights);


void
mha_fwd_kvcache_multiple(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                         at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                         at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                         c10::optional<at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                         c10::optional<at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                         at::Tensor &seqlens_k_list_, // batch_size
                         c10::optional<at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                         c10::optional<at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                         c10::optional<at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                         at::Tensor &block_table_list_, // batch_size x max_num_blocks_per_seq
                         c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                         at::Tensor &out_list_,             // batch_size x seqlen_q x num_heads x head_size
                         at::Tensor &out_es_sum_list_,             // batch_size x seqlen_q x num_heads x head_size
                         at::Tensor &out_es_min_list_,             // batch_size x seqlen_q x num_heads x head_size
                         at::Tensor &out_es_buffer_,          // batch_size x seqlen_q x num_heads x head_size
                         float softmax_scale,
                         bool is_causal,
                         int window_size_left,
                         int window_size_right,
                         bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                         int num_splits,
                         int num_local_tokens,
                         bool return_attn_weights,
                         float *es_acc,
                         float *es_min,
                         int *total_seq_lens,
                         int block_chunk_size,
                         float threshold,
                         std::atomic<int> &load_budget,
                         std::atomic<int> &compute_budget,
                         volatile int *seq_states,
                         volatile int *compute_iteration);
