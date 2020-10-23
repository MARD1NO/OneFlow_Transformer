# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer model helper methods."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math

import oneflow as flow
import oneflow.typing as tp
import numpy as np

_NEG_INF = -1e9


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

      Calculates the position encoding as a mix of sine and cosine functions with
      geometrically increasing wavelengths.
      Defined and formulized in Attention is All You Need, section 3.5.

      TODO: OneFlow has no op like `tf.range`, so i use numpy to instead, and give it to GlobalFunction

      TOO: We need a operator like `tf.range` to instead numpy

      Args:
        length: Sequence length.
        hidden_size: Size
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

      Returns:
        Tensor with shape [length, hidden_size]
    """
    position = np.arange(length).astype(np.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
    raise Exception("We still cannot use OneFlow to build `tf.matrix_band_part` ")


#     with tf.name_scope("decoder_self_attention_bias"):
#         valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
#         valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
#         decoder_bias = _NEG_INF * (1.0 - valid_locs)
#     return decoder_bias


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

       Args:
       x: int tensor with any shape
       padding_value: int value that

       Returns:
       float tensor with same shape as x containing values 0 or 1.
          0 -> non-padding, 1 -> padding
    """
    with flow.scope.namespace("padding"):
        return flow.cast(flow.math.equal(x, flow.constant_like(like=x, value=padding_value, dtype=x.dtype)),
                         dtype=flow.float32)


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

        Bias tensor that is added to the pre-softmax multi-headed attention logits,
        which has shape [batch_size, num_heads, length, length]. The tensor is zero at
        non-padding locations, and -1e9 (negative infinity) at padding locations.

        Args:
        x: int tensor with shape [batch_size, length]

        Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with flow.scope.namespace("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = flow.expand_dims(
            flow.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


# test
# if __name__ == "__main__":
#     @flow.global_function()
#     def multi_attention() -> tp.Numpy:
#         with flow.scope.namespace("multi"):
#
#             x = flow.get_variable("x",
#                                   shape=(10, 10),
#                                   initializer=flow.random_normal_initializer(mean=5.0),
#                                   )
#
#             out = get_padding_bias(x)
#
#             return out
#
#
#     check = flow.train.CheckPoint()
#     check.init()
#
#     out = multi_attention()
#     print(out.shape)
#     print(out)
