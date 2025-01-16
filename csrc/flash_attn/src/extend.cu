#include <cuda_runtime.h>
#include "flash.h"


__global__ 
void process_es_flash_kernel(
    float *softmax_lse, float *es,
    int *seq_lens,
    const int batch_size, const int num_heads, 
    const int max_num_blocks, const int block_size,
    float *es_acc, float *es_min,
    int *total_seq_lens, int block_chunk_size,
    volatile int *seq_states,
    int iteration, float threshold
) {
    const int seq_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int lane_idx = thread_idx % 32;
    const int warp_idx = thread_idx / 32;
    const int num_warps = blockDim.x / 32;
    const int num_blocks = seq_lens[seq_idx] / block_size;
    if (num_blocks <= 0) {
      return;
    }
    const int num_blocks_roundup = (num_blocks + 32 - 1) / 32 * 32;
    const int num_blocks_total = (total_seq_lens[seq_idx] + block_size - 1) / block_size;
    const int num_blocks_left = num_blocks_total - (iteration + 1) * block_chunk_size;
    if (num_blocks_left <= 0) {
      if (thread_idx == 0) {
        seq_states[seq_idx] = 0;
      }
      return;
    }
    extern __shared__ char shared_mem[];
    float *sm_ptg = reinterpret_cast<float*>(shared_mem);
    for (int head_idx = warp_idx; head_idx < num_heads; head_idx += num_warps) {
      const int input_es_offset = (seq_idx * num_heads + head_idx) * max_num_blocks;
      float min_es = 999999;
      for (int idx = lane_idx; idx < num_blocks_roundup; idx += 32) {
        float cur_min_es = (idx < num_blocks) ? es[input_es_offset + idx] : 999999;
#pragma unroll
        for (int offset = 32 / 2; offset > 0; offset /= 2) {
          cur_min_es = fminf(cur_min_es, __shfl_down_sync(0xffffffff, cur_min_es, offset));
        }
        min_es = fminf(cur_min_es, min_es);
      }
      const int es_offset = seq_idx * num_heads + head_idx;
      if (lane_idx == 0) {
        float total_es;
        if (iteration == 0) {
          total_es = expf(softmax_lse[es_offset]);
        } else {
          total_es = es_acc[es_offset] + expf(softmax_lse[es_offset]);
          min_es = fminf(es_min[es_offset], min_es);
        }
        es_acc[es_offset] = total_es;
        es_min[es_offset] = min_es;
        softmax_lse[es_offset] = expf(softmax_lse[es_offset]);
        float head_ptg = total_es / (total_es + min_es * num_blocks_left);
        sm_ptg[head_idx] = head_ptg;
      }
    }
    __syncthreads();
    if (thread_idx == 0) {
      float total_ptg = 0;
      for (int idx = 0; idx < num_heads; idx++) {
        total_ptg += sm_ptg[idx];
      }
      if (total_ptg / num_heads >= threshold) {
        seq_states[seq_idx] = 0;
      }
      // printf("iteration: %d, ptg: %f, threshold: %f\n", iteration, (total_ptg / num_heads), threshold);
    }
}

__global__ 
void proceed_kernel(volatile int *compute_iteration, int iteration) {
  atomicAdd((int *)compute_iteration, 1);
}

void process_es(Flash_fwd_params &params,
                float *es_acc, float *es_min,
                int *total_seq_lens, int block_chunk_size,
                volatile int *seq_states, volatile int *compute_iteration,
                int iteration, float threshold, cudaStream_t stream) {
    const int batch_size = params.b;
    const int num_heads = params.h * params.seqlen_q;
    const int max_num_blocks = params.attn_weights_cols;
    const int block_size = params.page_block_size;
    dim3 grid(batch_size);
    dim3 block(512);
    const int warp_size = 32;
    const int num_warps = 512 / warp_size;
    const int smem_size = num_heads * 4;
    process_es_flash_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<float*>(params.softmax_lse_ptr),
        reinterpret_cast<float*>(params.attn_weights_ptr),
        reinterpret_cast<int*>(params.cu_seqlens_k),
        batch_size, num_heads, max_num_blocks, block_size,
        es_acc, es_min, total_seq_lens, block_chunk_size,
        seq_states,
        iteration, threshold
    );
    proceed_kernel<<<1, 1, 0, stream>>>(compute_iteration, iteration);
}

__global__ 
void dummy_kernel(int iteration) {
  return;
}

void dummy(int iteration, cudaStream_t stream) {
  dummy_kernel<<<1, 1, 0, stream>>>(iteration);
}
