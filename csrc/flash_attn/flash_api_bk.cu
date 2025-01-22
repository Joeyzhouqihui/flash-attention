/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include "flash_api.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped=false,
                      int num_local_tokens=0,
                      int attn_scores_rows=0,
                      int attn_scores_cols=0,
                      void *attn_scores_d=nullptr,
                      bool reduce_attn_scores=false,
                      bool is_prefill=false,
                      int ngroups=1,
                      int attn_weights_rows=0,
                      int attn_weights_cols=0,
                      void *attn_weights_d=nullptr,
                      bool return_attn_weights=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    params.num_local_tokens = num_local_tokens;
    params.attn_scores_rows = attn_scores_rows;
    params.attn_scores_cols = attn_scores_cols;
    params.attn_scores_ptr = attn_scores_d;
    params.reduce_attn_scores = reduce_attn_scores;
    params.attn_weights_rows = attn_weights_rows;
    params.attn_weights_cols = attn_weights_cols;
    params.attn_weights_ptr = attn_weights_d;
    params.return_attn_weights = return_attn_weights;
    params.is_prefill = is_prefill;
    params.ngroups = ngroups;
    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            // if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
            //     run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            // } else {
            //     // printf("split\n");
            //     run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            // }
            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
        });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        // if (num_splits < 1) {
        //     // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
        //     params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
        // }
        // if (params.num_splits > 1) {
        //     at::Tensor softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
        //     at::Tensor out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
        //     params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        //     params.oaccum_ptr = out_accum.data_ptr();
        // }
        // TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
        params.num_splits = 1;
    }
}

void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

void
mha_fwd_kvcache(at::Tensor &old_q,                  // batch_size x seqlen_q x num_heads x head_size
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
                bool return_attn_weights) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto dprops = at::cuda::getCurrentDeviceProperties();

    at::Tensor q = old_q;
    auto q_dtype = q.dtype();
    const bool paged_KV = block_table_.has_value();
    at::Tensor block_table = block_table_.value();

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    const int seqlen_q_og = seqlen_q;
    int num_heads = sizes[2];
    const int num_heads_og = num_heads;
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int seqlen_k = max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = batch_size;
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        // CHECK_DEVICE(out);
        // TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, batch_size, seqlen_q_og, num_heads_og, head_size_og);
        // CHECK_SHAPE(out, batch_size, num_heads_og, head_size_og);
        if (head_size_og % 8 != 0) {
            out = torch::empty_like(q_padded);
        } else if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads, seqlen_q, head_size_og}).transpose(1, 2);
        }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor softmax_lse;
    if (out_es_sum_.has_value()) {
        softmax_lse = out_es_sum_.value();
        softmax_lse = softmax_lse.reshape({batch_size, num_heads, seqlen_q});
    } else {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    }

    at::Tensor attn_weights;
    if (return_attn_weights) {
        int attn_weights_rows = ngroups;
        int kBlockN = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
        int attn_weights_cols = round_multiple(seqlen_k, kBlockN) / page_block_size;
        TORCH_CHECK(kBlockN % page_block_size == 0, "kBlockN must be multiple of page size");
        
        if (out_es_buffer_.has_value()) {
            attn_weights = out_es_buffer_.value();
            attn_weights = attn_weights.reshape({batch_size, num_heads, attn_weights_rows, attn_weights_cols});
        } else {
            attn_weights = torch::empty({batch_size, num_heads, attn_weights_rows, attn_weights_cols}, opts.dtype(at::kFloat));
        }
        // attn_weights.fill_(999999);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     false,
                     num_local_tokens,
                     0,
                     0,
                     nullptr,
                     false,
                     false,
                     ngroups,
                     return_attn_weights ? attn_weights.sizes()[2] : 0,
                     return_attn_weights ? attn_weights.sizes()[3] : 0,
                     return_attn_weights ? attn_weights.data_ptr() : nullptr,
                     return_attn_weights);
                     

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        // TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        // TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        // TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        // TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        // TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        // CHECK_DEVICE(k); CHECK_DEVICE(v);
        // TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        // TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        // CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        // CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        // TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        // CHECK_DEVICE(seqlens_k);
        // CHECK_CONTIGUOUS(seqlens_k);
        // CHECK_SHAPE(seqlens_k, batch_size);
        params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());

    if (rotary_cos_.has_value()) {
        // TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        // CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        // TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        // TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        // TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        // CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        // CHECK_CONTIGUOUS(rotary_cos);
        // TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        // TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        // CHECK_DEVICE(rotary_sin);
        // CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        // CHECK_CONTIGUOUS(rotary_sin);
        // TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        // CHECK_DEVICE(cache_batch_idx);
        // CHECK_CONTIGUOUS(cache_batch_idx);
        // TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, /*dropout*/0.f, num_splits, dprops, opts);

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }
}

void
mha_fwd_kvcache_multiple(at::Tensor &old_q,                  // batch_size x seqlen_q x num_heads x head_size
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
                         volatile int *seq_states,
                         volatile int *compute_iteration_cnt,
                         volatile int *buffer_states) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto dprops = at::cuda::getCurrentDeviceProperties();

    at::Tensor q = old_q;
    auto q_dtype = q.dtype();
    const bool paged_KV = true;
    at::Tensor block_table = block_table_list_[0];

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    const int seqlen_q_og = seqlen_q;
    int num_heads = sizes[2];
    const int num_heads_og = num_heads;
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int seqlen_k = max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = batch_size;
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::Tensor out = out_list_[0];
    if (seqlenq_ngroups_swapped) {
        out = out.reshape({batch_size, num_heads, seqlen_q, head_size_og}).transpose(1, 2);
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor softmax_lse = out_es_sum_list_[0];
    softmax_lse = softmax_lse.reshape({batch_size, num_heads, seqlen_q});

    at::Tensor attn_weights;
    if (return_attn_weights) {
        int attn_weights_rows = ngroups;
        int kBlockN = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
        int attn_weights_cols = round_multiple(seqlen_k, kBlockN) / page_block_size;
        TORCH_CHECK(kBlockN % page_block_size == 0, "kBlockN must be multiple of page size");
        attn_weights = out_es_buffer_;
        attn_weights = attn_weights.reshape({batch_size, num_heads, attn_weights_rows, attn_weights_cols});
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, 
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     false,
                     num_local_tokens,
                     0,
                     0,
                     nullptr,
                     false,
                     false,
                     ngroups,
                     return_attn_weights ? attn_weights.sizes()[2] : 0,
                     return_attn_weights ? attn_weights.sizes()[3] : 0,
                     return_attn_weights ? attn_weights.data_ptr() : nullptr,
                     return_attn_weights);
                     

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        // TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        // TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        // TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        // TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        // TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        // CHECK_DEVICE(k); CHECK_DEVICE(v);
        // TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        // TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        // CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        // CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    auto seqlens_k = seqlens_k_list_[0];
    params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    params.is_seqlens_k_cumulative = false;

    if (rotary_cos_.has_value()) {
        // TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        // CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        // TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        // TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        // TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        // CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        // CHECK_CONTIGUOUS(rotary_cos);
        // TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        // TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        // CHECK_DEVICE(rotary_sin);
        // CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        // CHECK_CONTIGUOUS(rotary_sin);
        // TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        // CHECK_DEVICE(cache_batch_idx);
        // CHECK_CONTIGUOUS(cache_batch_idx);
        // TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, /*dropout*/0.f, num_splits, dprops, opts);

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    int max_iterations = block_table_list_.size(0);
    params.max_iterations = max_iterations;
    params.threshold = threshold;
    at::Tensor num_finish_seqs_tensor = torch::zeros({1}, opts.dtype(torch::kInt32));
    params.num_finish_seqs = reinterpret_cast<int*>(num_finish_seqs_tensor.data_ptr());
    params.es_acc = es_acc;
    params.es_min = es_min;
    params.total_seq_lens = total_seq_lens;
    params.block_chunk_size = block_chunk_size;
    params.seq_states = seq_states;
    params.buffer_states = buffer_states;
    params.compute_iteration_cnt = compute_iteration_cnt;
    at::Tensor barrier_tensor1 = torch::zeros({1}, opts.dtype(torch::kInt32));
    params.barrier1 = reinterpret_cast<int*>(barrier_tensor1.data_ptr());
    at::Tensor barrier_tensor2 = torch::zeros({1}, opts.dtype(torch::kInt32));
    params.barrier2 = reinterpret_cast<int*>(barrier_tensor2.data_ptr());
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        at::Tensor out = out_list_[iteration];
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads, seqlen_q, head_size_og}).transpose(1, 2);
        }
        params.o_ptr_list[iteration] = out.data_ptr();
        at::Tensor softmax_lse = out_es_sum_list_[iteration];
        softmax_lse = softmax_lse.reshape({batch_size, num_heads, seqlen_q});
        params.softmax_lse_ptr_list[iteration] = softmax_lse.data_ptr();
        at::Tensor seqlens_k = seqlens_k_list_[iteration];
        params.cu_seqlens_k_list[iteration] = reinterpret_cast<int*>(seqlens_k.data_ptr());
        at::Tensor block_table = block_table_list_[iteration];
        params.block_table_list[iteration] = reinterpret_cast<int*>(block_table.data_ptr());
    }
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
    cudaStreamSynchronize(stream);
    std::cout<<num_finish_seqs_tensor<<std::endl;
    printf("max iteration: %d, iteration: %d\n", max_iterations, (*compute_iteration_cnt));
}