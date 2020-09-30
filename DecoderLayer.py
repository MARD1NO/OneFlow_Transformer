"""
author: zzk
Build DecoderLayer
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np
from MultiheadAttention import MultiheadAttention
from positionwiseFeedForward import positionwiseFeedForward


class DecoderLayer(object):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Build DecoderLayer
        :param d_model: The dims of Model
        :param num_heads: The num of heads
        :param dff: The dim of FFN
        :param rate: The dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiheadAttention(d_model, num_heads)
        self.mha2 = MultiheadAttention(d_model, num_heads)

    def __call__(self,
                 x,
                 enc_output,
                 training=True,
                 look_ahead_mask=None,
                 padding_mask=None):
        """
        :param x: The previous layer output
        :param enc_output: The encoder output
        :param training: Whether training
        :param look_ahead_mask: The look ahead mask Blob
        :param padding_mask: The padding mask
        :return:
        """
        with flow.scope.namespace("MHA1"):
            attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
            if training:
                attn1 = flow.nn.dropout(attn1, rate=self.rate)
            out1 = flow.layers.layer_norm(attn1 + x, epsilon=1e-6)

        with flow.scope.namespace("MHA2"):
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
            if training:
                attn2 = flow.nn.dropout(attn2, rate=self.rate)
            out2 = flow.layers.layer_norm(attn2 + out1, epsilon=1e-6)

        with flow.scope.namespace("FFN"):
            ffn_output = positionwiseFeedForward(out2, self.d_model, self.dff)
            if training:
                ffn_output = flow.nn.dropout(ffn_output, rate=self.rate)
            out3 = flow.layers.layer_norm(ffn_output + out2, epsilon=1e-6)

        return out3, attn_weights_block1, attn_weights_block2


# Test
#
# if __name__ == "__main__":
#     @flow.global_function()
#     def decoder() -> tp.Numpy:
#         with flow.scope.namespace("multi"):
#             decoder_layer = DecoderLayer(512, 8, 2048)
#             x = flow.get_variable("x",
#                                   shape=(64, 50, 512),
#                                   initializer=flow.zeros_initializer())
#             encoder_layer_output = flow.get_variable("encoder_layer_output",
#                                                      shape=(64, 43, 512),
#                                                      initializer=flow.zeros_initializer())
#             out, _, __ = decoder_layer(x, encoder_layer_output)
#
#         return out
#
#     out = decoder()
#     print(out.shape)