"""
author: ZZK
Build Transformer
"""
import oneflow as flow
import oneflow.typing as tp
from Encoder import Encoder
from Decoder import Decoder
from new_positional_layer import positional_encoding
import numpy as np


def transformer(input_tensor,
                target_tensor,
                encoder_pos_encoding,
                decoder_pos_encoding,
                training,
                enc_padding_mask,
                look_ahead_mask,
                dec_padding_mask,
                num_layers,
                d_model,
                num_heads,
                dff,
                input_vocab_size,
                target_vocab_size,
                pe_input,
                pe_target,
                rate=0.1):
    """
    The params about input
    :param input_tensor: The input tensor
    :param target_tensor: The target tensor
    :param training: Whether training
    :param enc_padding_mask: The encoder padding mask
    :param look_ahead_mask: The look ahead mask
    :param dec_padding_mask: The decoder padding mask

    The params about Encoder and Decoder
    :param num_layers: The numbers of layers.
    :param d_model: The dims of model
    :param num_heads: The numbers of head
    :param dff: The dims of FFN
    :param input_vocab_size: The input vocab size
    :param target_vocab_size: The target vocab size
    :param pe_input: The position encode input
    :param pe_target: The position encode target
    :param rate: The dropout rate
    :return:
    """

    # # The positional encoding for Encoder
    # encoder_pos_encoding = positional_encoding(pe_input, d_model)
    # # The positional encoding for Decoder
    # decoder_pos_encoding = positional_encoding(pe_target, d_model)

    # Build Encoder
    encoder = Encoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      input_vocab_size=input_vocab_size,
                      maximum_position_encoding=pe_input,
                      rate=rate)
    # Build Decoder
    decoder = Decoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      target_vocab_size=target_vocab_size,
                      maximum_position_encoding=pe_target,
                      rate=rate)

    # Do the forward
    with flow.scope.namespace("Encoder output"):
        enc_output = encoder(x=input_tensor,
                             pos_encoding=encoder_pos_encoding,
                             training=training,
                             mask=enc_padding_mask)

    with flow.scope.namespace("Decoder output"):
        dec_output, attention_weights = decoder(x=target_tensor,
                                                pos_encoding=decoder_pos_encoding,
                                                enc_output=enc_output,
                                                training=training,
                                                look_ahead_mask=look_ahead_mask,
                                                padding_mask=dec_padding_mask)
    with flow.scope.namespace("Final output"):
        final_output = flow.layers.dense(inputs=dec_output,
                                         units=target_vocab_size,
                                         name="final_dense")
    return final_output, attention_weights


# test
def test(training,
         enc_padding_mask,
         look_ahead_mask,
         dec_padding_mask,
         num_layers,
         d_model,
         num_heads,
         dff,
         input_vocab_size,
         target_vocab_size,
         pe_input,
         pe_target):

    temp_input = np.ones(shape=(64, 38), dtype=np.int64)
    print(temp_input.shape)

    temp_output = np.ones(shape=(64, 36), dtype=np.int64)
    print(temp_output.shape)

    pos_en = positional_encoding(pe_input, d_model)
    print(pos_en.shape)

    pos_de = positional_encoding(pe_target, d_model)
    print(pos_de.shape)



    @flow.global_function()
    def testdecoder(x: tp.Numpy.Placeholder(shape=temp_input.shape, dtype=flow.int64),
                    y: tp.Numpy.Placeholder(shape=temp_output.shape, dtype=flow.int64),
                    pos_encode: tp.Numpy.Placeholder(shape=pos_en.shape, dtype=flow.float32),
                    pos_decode: tp.Numpy.Placeholder(shape=pos_de.shape, dtype=flow.float32)) -> tp.Numpy:
        with flow.scope.namespace("transformer"):
            out, _ = transformer(x, y, pos_encode, pos_decode, training,
                                 enc_padding_mask, look_ahead_mask, dec_padding_mask,
                                 num_layers, d_model, num_heads, dff,
                                 input_vocab_size, target_vocab_size,
                                 pe_input, pe_target)

        return out

    return testdecoder(temp_input, temp_output, pos_en, pos_de)


out = test(training=False,
           enc_padding_mask=None,
           look_ahead_mask=None,
           dec_padding_mask=None,
           num_layers=6,
           d_model=512,
           num_heads=8,
           dff=2048,
           input_vocab_size=8500,
           target_vocab_size=8000,
           pe_input=10000,
           pe_target=6000)
print(out.shape)