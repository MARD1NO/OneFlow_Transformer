"""
author: zzk
Build attention Layer
"""
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Build Attention Layer
    :param query: Query Matrix
    :param key: Key Matrix
    :param value: Value Matrix
    :param mask: The Mask
    :return:
    """
    matmul_qk = flow.matmul(query, key, transpose_b=True)

    # scaled matmul_qk
    d_k = flow.constant(query.shape[-1], dtype=flow.float32)
    scaled_attention_logits = matmul_qk / flow.math.sqrt(d_k)

    # Add mask
    if mask is not None:
        scaled_attention_logits += (mask * 1e-9)

    attention_weights = flow.nn.softmax(scaled_attention_logits, axis=-1)
    out = flow.matmul(attention_weights, value)

    return out, attention_weights


# # Test
if __name__ == "__main__":
    @flow.global_function()
    def test_attention() -> Tuple[tp.Numpy, tp.Numpy]:
        q = flow.get_variable("q",
                              # shape=(1, 512, 64),
                              shape=(1, 3),
                              initializer=flow.random_uniform_initializer(),
                              )
        k = flow.get_variable("k",
                              # shape=(1, 512, 64),
                              shape=(4, 3),
                              initializer=flow.random_uniform_initializer(),
                              )
        v = flow.get_variable("v",
                              # shape=(1, 512, 64),
                              shape=(4, 2),
                              initializer=flow.random_uniform_initializer(),
                              )
        out, att = scaled_dot_product_attention(q, k, v)
        return out, att


    check = flow.train.CheckPoint()
    check.init()
    out, att = test_attention()
    print(out.shape)
    print("Attention weights: ", att)
    print("Output is: ", out)
