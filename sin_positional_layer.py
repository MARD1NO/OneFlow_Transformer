"""
author: zzk

The sin Positional Encoder

TODO: Depreciated

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

    position = np.arange(0, max_len)
    position = np.expand_dims(position, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def generate_variable(pe_input: tp.Numpy.Placeholder(shape=pe.shape, dtype=flow.float32),
                          x_input: tp.Numpy.Placeholder(shape=(1, 100, 20), dtype=flow.float32)) -> tp.Numpy:

        zero_blob = flow.get_variable(name=name + "zero_Blob",
                                      shape=(1, pe.shape[0], pe.shape[1]),
                                      trainable=False,
                                      initializer=flow.zeros_initializer(),
                                      )
        pe_blob = zero_blob + pe_input
        pe_blob = flow.slice(pe_blob,
                             begin=[None, 0, None],
                             size=[None, x.shape[1], None],
                             name=name+"Slice")
        pe_blob += x_input
        return flow.nn.dropout(pe_blob, rate=dropout)

    # check_point = flow.train.CheckPoint()
    # check_point.init()

    position = generate_variable(pe, x)

    return position


# # test
#

# input_tensor = np.ones(shape=(1, 100, 20)).astype(np.float32)
# out = positionalEncoder(input_tensor, 20, 0)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(15, 5))
# plt.plot(np.arange(100), out[0, :, 4:8])
# plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
# plt.show()
