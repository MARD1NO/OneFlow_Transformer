"""
author: zzk
positional layer
"""

import oneflow as flow
import oneflow.typing as tp
import numpy as np
import math
from matplotlib import pyplot as plt


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.expand_dims(angle_rads, axis=0).astype(np.float32)

    return pos_encoding


@flow.global_function()
def pos_encode(x:tp.Numpy.Placeholder(shape=(1, 50, 512), dtype=flow.float32)) -> tp.Numpy:
    zero = flow.get_variable(name="zero",
                             shape=(1, 50, 512),
                             dtype=flow.float32,
                             initializer=flow.zeros_initializer(),
                             trainable=False)
    out = x + zero
    return out


# if __name__ == "__main__":
#     check_point = flow.train.CheckPoint()
#     check_point.init()
#     pos = positional_encoding(50, 512)
#     out = pos_encode(pos)
#     print(out.shape)
#     # pos_encoding = positional_encoding(50, 512)
#     # print(pos_encoding.shape)
#     plt.pcolormesh(out[0], cmap='RdBu')
#     plt.xlabel('Depth')
#     plt.xlim((0, 512))
#     plt.ylabel('Position')
#     plt.colorbar()
#     plt.show()