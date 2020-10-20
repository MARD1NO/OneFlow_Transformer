"""
author: zzk

Build a positionwise FeedForward Layer
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np


def positionwiseFeedForward(x, d_model, d_ff, name="FFN_layer"):
    """
    Build Positionwise FeedForward Layer

    FFN(x) = max(0, W1*X+B1)*W2 + B2

    :param x: The input Blob
    :param d_model: The channels of input
    :param d_ff: The channels of hidden
    :param dropout: The dropout probability
    :return:
    """
    initializer_range = 0.2
    init = flow.truncated_normal(initializer_range)
    regularizer_range = 0.0005
    reg = flow.regularizers.l2(regularizer_range)
    x = flow.layers.dense(inputs=x,
                          units=d_ff,
                          kernel_initializer=init,
                          kernel_regularizer=reg,
                          bias_initializer=init,
                          bias_regularizer=reg,
                          name=name+"_W1")
    
    x = flow.layers.dense(inputs=x,
                          units=d_model,
                          kernel_initializer=init,
                          kernel_regularizer=reg,
                          bias_initializer=init,
                          bias_regularizer=reg,
                          name=name+"_W2")
    return x


# Test
# @flow.global_function()
# def test_FFN(x: tp.Numpy.Placeholder(shape=(1, 4), dtype=flow.float32)) -> tp.Numpy:
#     out = positionwiseFeedForward(x, 4, 10)
#     return out
#
#
# check = flow.train.CheckPoint()
# check.init()
#
#
# x = np.array([[0, 2, 0, 5]]).astype(np.float32)
# out = test_FFN(x)
# print(out.shape)