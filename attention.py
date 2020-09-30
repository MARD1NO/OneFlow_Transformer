"""
author: zzk
Build attention Layer
"""
import oneflow as flow
import oneflow.typing as tp


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Build Attention Layer
    :param query: Query Matrix
    :param key: Key Matrix
    :param value: Value Matrix
    :param mask: The Mask
    :return:
    """
    d_k = flow.constant(query.shape[-1], dtype=flow.float32)

    scores = flow.matmul(query, key, transpose_b=True) * flow.math.rsqrt(d_k)

    # Add mask
    if mask is not None:
        scores += (mask * 1e-9)

    p_attention = flow.nn.softmax(scores, axis=-1)
    out = flow.matmul(p_attention, value)

    return out, p_attention

# # Test
# @flow.global_function()
# def test_attention() -> tp.Numpy:
#     q = flow.get_variable("q",
#                           # shape=(1, 512, 64),
#                           shape=(1, 3),
#                           initializer=flow.zeros_initializer(),
#                           )
#     k = flow.get_variable("k",
#                           # shape=(1, 512, 64),
#                           shape=(4, 3),
#                           initializer=flow.zeros_initializer(),
#                           )
#     v = flow.get_variable("v",
#                           # shape=(1, 512, 64),
#                           shape=(4, 2),
#                           initializer=flow.zeros_initializer(),
#                           )
#     out, att = scaled_dot_product_attention(q, k, v)
#     return out
#
# check = flow.train.CheckPoint()
# check.init()
# out = test_attention()
# print(out.shape)