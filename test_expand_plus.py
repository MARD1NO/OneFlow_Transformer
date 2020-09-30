"""
BUG
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np
import math


def positionalEncoder(x, d_model, dropout, max_len=5000, name="Positional_"):
    """
    The positional Encoder Layer
    :param x: The input Blob.
    :param d_model: The dim of model.
    :param max_len: The max length of sequence.
    :param dropout: The dropout rate.
    :return:
    """
    pe = np.zeros(shape=(max_len, d_model), dtype=np.float32)


    @flow.global_function()
    def generate_variable(pe_input: tp.Numpy.Placeholder(shape=pe.shape, dtype=flow.float32),
                          x_input: tp.Numpy.Placeholder(shape=(1, 100, 20), dtype=flow.float32)) -> tp.Numpy:

        zero_blob = flow.get_variable(name=name + "zero_Blob",
                                      shape=(1, pe.shape[0], pe.shape[1]),
                                      initializer=flow.zeros_initializer()
                                      )
        pe_blob = zero_blob + pe_input
        pe_blob = flow.slice(pe_blob,
                             begin=[None, 0, None],
                             size=[None, x.shape[1], None],
                             name=name+"Slice")
        pe_blob += x_input
        return pe_blob

    check_point = flow.train.CheckPoint()
    check_point.init()

    position = generate_variable(pe, x)

    return position

input_tensor = np.ones(shape=(1, 100, 20)).astype(np.float32)
out = positionalEncoder(input_tensor, 20, 0)