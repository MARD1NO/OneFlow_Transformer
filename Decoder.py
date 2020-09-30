"""
author: zzk
Build Decoder
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np
from DecoderLayer import DecoderLayer
from embedding_layer import EmbeddingLayer
from new_positional_layer import positional_encoding


class Decoder(object):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        """
        Build Encoder
        :param num_layers: The numbers of Encoder Layer
        :param d_model: The dim of model
        :param num_heads: The numbers of head
        :param dff: The dim of FFN
        :param target_vocab_size: The target Vocab size
        :param maximum_position_encoding: The maximum position encoding
        :param rate: The dropout rate
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding

        # Build Multi encoder layers
        self._build_multi_decoder_layer(d_model, num_heads, dff, rate)

    def _build_multi_decoder_layer(self, d_model, num_heads, dff, rate):
        """
        Build multi decoder layers
        :param d_model: The dim of model
        :param num_heads: The numbers of head
        :param dff: The dim of FFN
        :param rate: The dropout rate
        :return: A Encoder layers List
        """
        self.dec_layers = []
        for i in range(self.num_layers):
            with flow.scope.namespace('Decoder_{}'.format(i)):
                self.dec_layers.append(DecoderLayer(d_model=d_model,
                                                    num_heads=num_heads,
                                                    dff=dff,
                                                    rate=rate))

    def __call__(self, x, pos_encoding, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward
        :param x: The input X
        :param pos_encoding: The positional encoding
        :param enc_output: The encoder output
        :param training: Whether training
        :param look_ahead_mask: The look ahead mask
        :param padding_mask: The padding mask
        :return:
        """
        # Sequence length
        seq_len = x.shape[1]
        attention_weights = {}

        # Embedding
        with flow.scope.namespace("Decoder_Embedding"):
            x = EmbeddingLayer(x,
                               vocab_size=self.target_vocab_size,
                               embedding_size=self.d_model)
            d_model_constant = flow.constant(self.d_model,
                                             dtype=flow.float32,
                                             shape=(1,))
            x *= flow.math.sqrt(d_model_constant)
            print(x.shape)

        # Position encoding
        with flow.scope.namespace("Decoder_Position_encoding"):
            pos_encoding = flow.slice(pos_encoding,
                                      begin=[None, 0, None],
                                      size=[None, seq_len, None])
            x += pos_encoding
            if training:
                x = flow.nn.dropout(x,
                                    rate=self.rate)


        # Decoding
        with flow.scope.namespace("Decoder_Multi_decoder"):
            for i in range(self.num_layers):
                with flow.scope.namespace('decoder_{}'.format(i)):
                    x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

                    attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
                    attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


# test
# def test(num_layers=2,
#          d_model=512,
#          num_heads=8,
#          dff=2048,
#          target_vocab_size=8500,
#          maximum_position_encoding=10000):
#
#     x_ = np.ones(shape=(64, 62), dtype=np.int64)
#     print(x_.shape)
#
#     pos = positional_encoding(maximum_position_encoding, d_model)
#     print(pos.shape)
#
#     enc_output = np.ones(shape=(64, 62, 512), dtype=np.float32)
#     print(enc_output.shape)
#
#     @flow.global_function()
#     def testdecoder(x: tp.Numpy.Placeholder(shape=(64, 62), dtype=flow.int64),
#                     pos_encode: tp.Numpy.Placeholder(shape=pos.shape, dtype=flow.float32),
#                     enc: tp.Numpy.Placeholder(shape=enc_output.shape, dtype=flow.float32)) -> tp.Numpy:
#         with flow.scope.namespace("decoder"):
#             decoder = Decoder(num_layers=num_layers,
#                               d_model=d_model,
#                               num_heads=num_heads,
#                               dff=dff,
#                               target_vocab_size=target_vocab_size,
#                               maximum_position_encoding=maximum_position_encoding)
#             out, attn = decoder(x, pos_encode, enc, training=False,
#                                 look_ahead_mask=None,
#                                 padding_mask=None)
#
#         return out
#
#     return testdecoder(x_, pos, enc_output)
#
#
# out = test()
# print(out.shape)