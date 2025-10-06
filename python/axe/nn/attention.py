from .module import Module
from .. import matmul, sqrt, array
from ..functional import softmax

class Attention(Module):
    """
    Implements scaled dot-product attention.
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout

    def __call__(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (N, L, E)
            key: Key tensor of shape (N, S, E)
            value: Value tensor of shape (N, S, V)
            mask: Optional mask tensor
        """
        E = query.shape[-1]

        # (N, L, E) @ (N, E, S) -> (N, L, S)
        key_t = key.transpose(-2, -1)
        scores = query.bmm(key_t) / sqrt(array(E))

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)

        # (N, L, S) @ (N, S, V) -> (N, L, V)
        output = attn_weights.bmm(value)
        return output, attn_weights

    def __repr__(self):
        return f"Attention(dropout={self.dropout})"