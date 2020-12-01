"""
author: zzk

The sin Positional Encoder

"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np


def get_angles(pos, i, d_model):
    """
    Compute angles

    The equation is  1 / 10000^(2i / d_model)
    :param pos: The position dims, shape=(position, 1)
    :param i: The d_model index, shape = (1, d_model)
    :param d_model: The hidden dims, int value
    :return:
    """
    # Get constant value as d_model
    d_model_constant = flow.constant(d_model, dtype=flow.float32, shape=(1,), name="One_constant")

    constant_10000 = flow.constant(10000, dtype=flow.float32, shape=(1, d_model), name="constant_10000")

    constant_2 = flow.constant_scalar(2, dtype=flow.float32)

    # Compute angle_rates = 1 / 10000^(2i / d_model)

    angle_rates = 1 / flow.math.pow(constant_10000,
                                    (constant_2 * flow.math.floor(i / constant_2)) / d_model_constant)

    return pos * angle_rates


def positional_encoding(position, d_model, name="positional_encoding"):
    """
    Do positional encoding
    :param position: The position
    :param d_model: The hidden dimension in model
    :return: shape like (1, position, d_model)
    """
    with flow.scope.namespace(name):
        # shape = (position, 1)
        input_pos = flow.expand_dims(flow.range(position, dtype=flow.float32, name="pos"), axis=1)

        # shape = (1, d_model)
        input_d_model = flow.expand_dims(flow.range(d_model, dtype=flow.float32, name="d_model"), axis=0)

        angle_rads = get_angles(input_pos, input_d_model, d_model)

        # Get a even range like (0, 2, 4, 6, ....., d_model)
        even_range = flow.range(0, d_model, 2, dtype=flow.int32, name="even_range")

        # Do the sin in even indexes
        even_out = flow.math.sin(flow.gather(angle_rads, even_range, axis=1))

        # Get a odd range like (1, 3, 5, 7, ....., d_model)
        odd_range = flow.range(1, d_model, 2, dtype=flow.int32, name="odd_range")

        # Do the cos in odd indexes
        odd_out = flow.math.cos(flow.gather(angle_rads, odd_range, axis=1))

        # Initialize Position encode constant
        position_encode = flow.constant(0, dtype=flow.float32, shape=(d_model, position), name="pos_ende")

        # Due to the scatter only support row indexes, we need to transpose
        even_out = flow.tensor_scatter_nd_update(position_encode,
                                                 flow.expand_dims(even_range, axis=1),
                                                 flow.transpose(even_out, perm=[1, 0]))

        odd_out = flow.tensor_scatter_nd_update(position_encode,
                                                flow.expand_dims(odd_range, axis=1),
                                                flow.transpose(odd_out, perm=[1, 0]))

        # Add even indexes value and odd indexes value
        out = even_out + odd_out

        # Because We have transposed in even_out and odd_out, So we need to transpose back
        out = flow.transpose(out, perm=[1, 0])
        # Expand dims in dim=0, we get shape like (1, position, d_model)
        out = flow.expand_dims(out, axis=0)

    return out

# # test
# if __name__ == "__train__":
#     @flow.global_function(type="train")
#     def test_postional_encoding() -> tp.Numpy:
#         pos = positional_encoding(50, 512)
#
#         sliced_pos = flow.slice(pos, [None, 2, None], [None, 4, None])
#
#         x_var = flow.get_variable(name="x",
#                                   dtype=flow.float32,
#                                   initializer=flow.ones_initializer(),
#                                   shape=(1, 50, 512))
#         out = x_var + pos
#         flow.optimizer.SGD(
#             flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
#         ).minimize(out)
#
#         # return pos
#         return sliced_pos
#
#     out = test_postional_encoding()
#
#     print(out.shape)
#     print(out)
