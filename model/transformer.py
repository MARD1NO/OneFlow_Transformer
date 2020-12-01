"""
author: ZZK
Build Transformer
"""
import oneflow as flow
import oneflow.typing as tp
from oneflow_transformer.model.Encoder import Encoder
from oneflow_transformer.model.Decoder import Decoder
import numpy as np


class Transformer:
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        # self.final_layer = Dense(target_vocab_size)
        self.target_vocab_size = target_vocab_size
        self.name = "Transformer"

    def __call__(self, inp, tar, training, enc_padding_mask,
                 look_ahead_mask, dec_padding_mask):
        with flow.scope.namespace(self.name + "_Encoder"):
            enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        with flow.scope.namespace(self.name + "_Decoder"):
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = flow.layers.dense(dec_output, units=self.target_vocab_size, name=self.name + "_final_layer")

        return final_output, attention_weights


# if __name__ == "__main__":
#     @flow.global_function(type="train")
#     def test_transformer(x: tp.Numpy.Placeholder(shape=(64, 62), dtype=flow.int64),
#                          y: tp.Numpy.Placeholder(shape=(64, 26), dtype=flow.int64)):
#         sample_transformer = Transformer(
#             num_layers=2, d_model=512, num_heads=8, dff=2048,
#             input_vocab_size=8500, target_vocab_size=8000,
#             pe_input=10000, pe_target=6000)
#
#         fn_out, _ = sample_transformer(x, y, training=False,
#                                        enc_padding_mask=None,
#                                        look_ahead_mask=None,
#                                        dec_padding_mask=None)
#         x_var = flow.get_variable(shape=(64, 26, 8000), dtype=flow.float32,
#                                   name="x_var", initializer=flow.random_normal_initializer())
#         loss = x_var + fn_out
#         with flow.scope.placement("gpu", "0:0"):
#             flow.optimizer.SGD(
#                 flow.optimizer.PiecewiseConsntScheduler([], [1e-3]), momentum=0
#             ).minimize(loss)
#         print("Final out shape is: ", fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
#
#
#     temp_input = np.ones((64, 62), dtype=np.int64)
#     temp_target = np.ones((64, 26), dtype=np.int64)
#
#     test_transformer(temp_input, temp_target)
