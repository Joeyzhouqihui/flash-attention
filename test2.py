import torch
import torch.nn.functional as F

def masked_multi_head_attention(query, key, value, mask=None):
    """
    Perform multi-head attention with different number of heads in query and key/value.

    Args:
        query (Tensor): (batch_size, seq_len_q, num_heads_q, head_dim).
        key (Tensor): (batch_size, seq_len_k, num_heads_k, head_dim).
        value (Tensor): (batch_size, seq_len_k, num_heads_k, head_dim).
        mask (Tensor): (batch_size, 1, seq_len_q, seq_len_k) for preventing attention to future tokens.

    Returns:
        Tuple[Tensor, Tensor]: Output and attention scores of the attention mechanism.
    """
    # Dimensions
    batch_size, seq_len_q, num_heads_q, head_dim = query.shape
    _, seq_len_k, num_heads_k, _ = key.shape

    # Ensure num_heads_q is a multiple of num_heads_k
    factor = num_heads_q // num_heads_k
    key = key.repeat(1, 1, factor, 1)
    value = value.repeat(1, 1, factor, 1)

    # Transpose key for proper matrix multiplication
    key = key.transpose(-2, -1)  # Now shape: (batch_size, head_dim, num_heads_k * factor, seq_len_k)

    # Calculate the dot product between query and transposed key
    scores = torch.matmul(query, key) / (head_dim ** 0.5)

    # Apply mask
    if mask is not None:
        scores += mask.unsqueeze(1)  # Broadcast mask across all heads

    # Softmax to normalize the scores
    attention_scores = F.softmax(scores, dim=-1)

    # Multiply by value
    output = torch.matmul(attention_scores, value)

    return output, attention_scores

# Dummy inputs
batch_size = 1
seq_len_q = 5
seq_len_k = 10
num_heads_q = 8
num_heads_k = 4
head_dim = 64

query = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim)
key = torch.randn(batch_size, seq_len_k, num_heads_k, head_dim)
value = torch.randn(batch_size, seq_len_k, num_heads_k, head_dim)
mask = torch.triu(torch.ones(seq_len_q, seq_len_k) * float('-inf'), diagonal=1).unsqueeze(0)

output, attention_scores = masked_multi_head_attention(query, key, value, mask)
