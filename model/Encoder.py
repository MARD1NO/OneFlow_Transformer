"""
author: zzk
Build Encoder
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np
# from oneflow_transformer.model.EncoderLayer import EncoderLayer
from EncoderLayer import EncoderLayer
# from oneflow_transformer.model.embedding_layer import EmbeddingLayer
from embedding_layer import EmbeddingLayer
# from oneflow_transformer.model.sin_positional_layer import positional_encoding
from sin_positional_layer import positional_encoding


class Encoder(object):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        """
        Build Encoder
        :param num_layers: The numbers of Encoder Layer
        :param d_model: The dim of model
        :param num_heads: The numbers of head
        :param dff: The dim of FFN
        :param input_vocab_size: The input Vocab size
        :param maximum_position_encoding: The maximum position encoding
        :param rate: The dropout rate
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model, name="Encoder_pos_encoding")

        # Build Multi encoder layers
        self._build_multi_encoder_layer(d_model, num_heads, dff, rate)

    def _build_multi_encoder_layer(self, d_model, num_heads, dff, rate):
        """
        Build multi encoder layers
        :param d_model: The dim of model
        :param num_heads: The numbers of head
        :param dff: The dim of FFN
        :param rate: The dropout rate
        :return: A Encoder layers List
        """
        self.enc_layers = []
        for i in range(self.num_layers):
            with flow.scope.namespace('Encoder_{}'.format(i)):
                self.enc_layers.append(EncoderLayer(d_model=d_model,
                                                    num_heads=num_heads,
                                                    dff=dff,
                                                    rate=rate))

    def __call__(self, x, training, mask):
        # Sequence length
        seq_len = x.shape[1]

        # Embedding
        with flow.scope.namespace("Encoder_Embedding"):
            x = EmbeddingLayer(x,
                               vocab_size=self.vocab_size,
                               embedding_size=self.d_model)
            d_model_constant = flow.constant_scalar(value=self.d_model, dtype=flow.float32, name="d_model_constant")
            x *= flow.math.sqrt(d_model_constant)

        # Position encoding
        with flow.scope.namespace("Encoder_Position_encoding"):
            # equal to self.pos_encoding[:, :seq_len, :]
            pos_encoding = flow.slice(self.pos_encoding,
                                      begin=[None, 0, None],
                                      size=[None, seq_len, None])
            x += pos_encoding
            if training:
                x = flow.nn.dropout(x,
                                    rate=self.rate)

        # Encoding
        with flow.scope.namespace("Encoder_Multi_encoder"):
            for i in range(self.num_layers):
                with flow.scope.namespace('encoder_{}'.format(i)):
                    x = self.enc_layers[i](x, training, mask)

        return x


if __name__ == "__main__": 
    @flow.global_function()
    def testencoder(x: tp.Numpy.Placeholder(shape=(64, 62), dtype=flow.int64)) -> tp.Numpy:
        with flow.scope.namespace("encoder"):
            encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
                            dff=2048, input_vocab_size=8500,
                            maximum_position_encoding=10000)
            out = encoder(x, training=False, mask=None)

        return out

    check = flow.train.CheckPoint()
    check.init()

    x = np.random.randint(0, 50, size=64*62, dtype=np.int64).reshape(64, 62)

    out = testencoder(x)
    print("Out shape is", out.shape) # -> (64, 62, 512)