"""
author: zzk
Build MultiHeadAttention Layer
"""

from oneflow_transformer.model.attention import scaled_dot_product_attention
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple


class MultiheadAttention(object):
    def __init__(self, d_model, num_heads):
        """
        Build Multihead Attention Layer
        :param d_model: The dims of model
        :param num_heads: The number of heads
        """
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

    def _split_heads(self, x, batch_size):
        """
        Split the Last Dimension into (num_heads, depth)
        Then Transpose to the shape as (batch_size, num_heads, seq_len, depth)
        :param x: The input Blob.
        :param batch_size: The batch size.
        :return:
        """
        x = flow.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return flow.transpose(x, perm=[0, 2, 1, 3])

    def _build_dense(self, x, unit, name="dense_"):
        """
        Build Dense Layer
        :param x:
        :return:
        """
        self.init_range = 0.2
        self.init = flow.truncated_normal_initializer(self.init_range)
        self.reg = flow.regularizers.l2(0.01)

        return flow.layers.dense(x,
                                 units=unit,
                                 kernel_initializer=self.init,
                                 kernel_regularizer=self.reg,
                                 bias_initializer=flow.zeros_initializer(),
                                 bias_regularizer=self.reg,
                                 name=name+"w")

    def __call__(self, v, k, q, mask):
        """
        Compute the Multi head attention
        :param v: The matrix V
        :param k: The matrix K
        :param q: The matrix Q
        :param mask: The Mask
        :return:
        """
        batch_size = q.shape[0]

        # Linear Layer
        q = self._build_dense(q, self.d_model, name="Dense_q_")  # (batch, seq_len, d_model)
        k = self._build_dense(k, self.d_model, name="Dense_k_")  # (batch, seq_len, d_model)
        v = self._build_dense(v, self.d_model, name="Dense_v_")  # (batch, seq_len, d_model)

        # Split Heads
        q = self._split_heads(q, batch_size)  # (batch, num_heads, seq_len, depth)
        k = self._split_heads(k, batch_size)  # (batch, num_heads, seq_len, depth)
        v = self._split_heads(v, batch_size)  # (batch, num_heads, seq_len, depth)

        # Compute Scaled attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = flow.transpose(scaled_attention,
                                          perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, depth)
        concat_attention = flow.reshape(scaled_attention,
                                        shape=(batch_size, -1, self.d_model))

        output = self._build_dense(concat_attention, unit=self.d_model, name="Dense_out_")

        return output, attention_weights


# test
# if __name__ == "__main__":
#     @flow.global_function()
#     def multi_attention() -> Tuple[tp.Numpy, tp.Numpy]:
#         with flow.scope.namespace("multi"):
#             mha = MultiheadAttention(512, 8)
#             x = flow.get_variable("x",
#                                   shape=(1, 60, 512),
#                                   initializer=flow.zeros_initializer(),
#                                   )
#             y = flow.get_variable("y",
#                                   shape=(1, 60, 512),
#                                   initializer=flow.zeros_initializer(),
#                                   )
#             z = flow.get_variable("z",
#                                   shape=(1, 60, 512),
#                                   initializer=flow.zeros_initializer(),
#                                   )
#             out, attn = mha(x, y, z, mask=None)
#
#             return out, attn
#
#
#     out, attn = multi_attention()
#     print(out.shape)
#     print(attn.shape)