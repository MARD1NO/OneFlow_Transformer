"""
author: zzk
Build EncoderLayer
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np
# from oneflow_transformer.model.MultiheadAttention import MultiheadAttention
from MultiheadAttention import MultiheadAttention
# from oneflow_transformer.model.positionwiseFeedForward import positionwiseFeedForward
from positionwiseFeedForward import positionwiseFeedForward


class EncoderLayer(object):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="EncoderLayer_"):
        self.mha = MultiheadAttention(d_model, num_heads)
        self.dff = dff
        self.d_model = d_model
        self.rate = rate
        self.name = name

    def __call__(self, x, training=True, mask=None):
        with flow.scope.namespace("MHA1"):
            attn_output, _ = self.mha(x, x, x, mask)
            if training:
                attn_output = flow.nn.dropout(attn_output, rate=self.rate)
            out1 = flow.layers.layer_norm(attn_output + x,
                                          epsilon=1e-6)

        with flow.scope.namespace("FFN1"):
            ffn_output = positionwiseFeedForward(out1, self.d_model, self.dff)

            if training:
                ffn_output = flow.nn.dropout(ffn_output, rate=self.rate)

            out2 = flow.layers.layer_norm(out1+ffn_output,
                                          epsilon=1e-6)

        return out2



# Test
# if __name__ == "__main__":
#     @flow.global_function()
#     def encoder() -> tp.Numpy:
#         with flow.scope.namespace("multi"):
#             encoder_layer = EncoderLayer(512, 8, 2048)
#             x = flow.get_variable("x",
#                                   shape=(64, 43, 512),
#                                   initializer=flow.zeros_initializer())
#             out = encoder_layer(x)

#         return out

#     out = encoder()
#     print(out.shape)