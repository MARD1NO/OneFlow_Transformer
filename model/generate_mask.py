import oneflow as flow
import oneflow.typing as tp
import numpy as np


def create_padding_mask(seq, name="CreatePad"):
    """
    Create padding mask
    :param seq: input sequence, shape=(batch, seq_lenth)
    :return:
    """
    with flow.scope.namespace(name):
        seq = flow.cast(flow.math.equal(seq, flow.constant_scalar(0,
                                                                  dtype=flow.int64,
                                                                  name="zero_mask_scalar")), flow.float32)
        # Expand dims from (a, b) -> (a, 1, 1, b)
        seq = flow.expand_dims(seq, axis=1)
        seq = flow.expand_dims(seq, axis=1)

    return seq


def create_look_ahead_mask(size):
    """
    Return a mask like
    [[0., 1., 1.],
     [0., 0., 1.],
     [0., 0., 0.]]
    :param size: The matrix size
    :return: look ahead mask
    """
    ones_blob = flow.get_variable(name="ones_blob",
                                  shape=[size, size],
                                  dtype=flow.float32,
                                  initializer=flow.ones_initializer(),
                                  trainable=False)
    mask = 1 - flow.math.tril(ones_blob, 0)
    return mask


def create_masks(inp, tar):
    """
    Creating the mask used in Transformer
    :param inp: The input sequence
    :param tar: The target sequence
    :return: Three mask
    """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, name="Create encoder pad")

    # Decoder padding mask
    dec_padding_mask = create_padding_mask(inp, name="Create decoder pad")

    # Look ahead mask
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar, name="Create target decoder pad")
    # TODO: May have backward Error ?
    combined_mask = flow.math.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

# test
# if __name__ == "__main__":
#
#     @flow.global_function()
#     def test_pad_mask(x: tp.Numpy.Placeholder(shape=(5, 256), dtype=flow.int64)) -> tp.Numpy:
#         out = create_padding_mask(x)
#         # out = create_look_ahead_mask(5)
#         return out
#
#
#     checkpoint = flow.train.CheckPoint()
#     checkpoint.init()
#     x = np.random.randint(0, 2500, size=(5, 256), dtype=np.int64)
#     a = test_pad_mask(x)
#     print(a)
