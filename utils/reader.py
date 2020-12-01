from oneflow_transformer.utils import tokenizer_v3
import oneflow as flow
import os
import numpy as np
from oneflow_transformer.model.generate_mask import *
from oneflow_transformer.model.transformer import Transformer
from oneflow_transformer.model.model_params import PathParams

# _TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
# _TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
# VOCAB_FILE = "vocab.ende.%d" % _TARGET_VOCAB_SIZE
#
# # _train_data_dir = "/home/zzk/Code/oneflow_transformer/model/raw_data/"
# _train_data_dir = "/home/zzk/Code/oneflow_transformer/model/raw_data_mini/"
#
#
# _TRAIN_DATA_SOURCES = [
#     {
#         "url": "http://data.statmt.org/wmt17/translation-task/"
#                "training-parallel-nc-v12.tgz",
#         "input": "news-commentary-v12.de-en.en",
#         "target": "news-commentary-v12.de-en.de",
#     },
#     {
#         "url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
#         "input": "commoncrawl.de-en.en",
#         "target": "commoncrawl.de-en.de",
#     },
#     {
#         "url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
#         "input": "europarl-v7.de-en.en",
#         "target": "europarl-v7.de-en.de",
#     },
# ]
#
# _raw_file_input_dir = [os.path.join(_train_data_dir, _data["input"])
#                        for _data in _TRAIN_DATA_SOURCES]
#
# _raw_file_target_dir = [os.path.join(_train_data_dir, _data["target"])
#                        for _data in _TRAIN_DATA_SOURCES]
#
# _raw_file_dir = _raw_file_input_dir + _raw_file_target_dir
#
# _data_dir = "/home/zzk/Code/oneflow_transformer/model/translate_ende/"
# _vocab_dir = os.path.join(_data_dir, "vocab.ende.32768")
#
# _max_length = 256
# _batch_size = 16

params = PathParams()

subtokenizer = tokenizer_v3.Subtokenizer.init_from_files(
    params.vocab_dir, params.raw_file_dir, params.TARGET_VOCAB_SIZE, params.TARGET_THRESHOLD,
    min_count=None)


def loss_function(real, pred):
    mask = flow.math.not_equal(real, flow.constant_scalar(0, dtype=flow.int64, name="zero constant"))

    real = flow.cast(real, dtype=flow.int32, name="cast_to_int32")
    loss_ = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)

    mask = flow.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return flow.math.reduce_mean(loss_)


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp, name="Create encoder pad")

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp, name="Create decoder pad")

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar, name="Create target decoder pad")
    # TODO: May have backward Error ?
    combined_mask = flow.math.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def read_data(raw_file_input_dir, raw_file_target_dir, subtokenizer, max_length, batch_num):
    with open(raw_file_input_dir) as f:
        _input_total_files = f.readlines()

    with open(raw_file_target_dir) as f:
        _target_total_files = f.readlines()

    # Compute the minimuim file length
    _min_file_length = min(len(_input_total_files), len(_target_total_files))

    def reader():
        _batch_num = 0
        _batch_input_seq = []
        _batch_target_seq = []
        for index in range(_min_file_length):

            _input_seq = _input_total_files[index]
            # print("Input sequence is: ", _input_seq)
            _target_seq = _target_total_files[index]
            # print("Target sequence is: ", _target_seq)

            # Ignore the sequence which length > max_length - 1 (Because we have a EOS_ID)
            if len(_input_seq) > max_length - 1 or len(_target_seq) > max_length - 1:
                continue

            # Encode the input sequence
            _encoded_input_seq = subtokenizer.encode(_input_seq, add_eos=True, max_length=256)
            _encoded_input_seq = np.array(_encoded_input_seq)

            # Encode the target sequence
            _encoded_target_seq = subtokenizer.encode(_target_seq, add_eos=True, max_length=256)
            _encoded_target_seq = np.array(_encoded_target_seq)

            _batch_input_seq.append(_encoded_input_seq)
            _batch_target_seq.append(_encoded_target_seq)
            _batch_num += 1

            if _batch_num == batch_num:
                # Generate a batch data
                yield np.array(_batch_input_seq), np.array(_batch_input_seq)
                # Reset
                _batch_num = 0
                _batch_input_seq = []
                _batch_target_seq = []

    return reader


if __name__ == "__main__":
    train_file_num = len(params.raw_file_input_dir)
    for train_file_id in range(train_file_num):
        reader = read_data(params.raw_file_input_dir[train_file_id],
                           params.raw_file_target_dir[train_file_id],
                           subtokenizer,
                           params.max_length,
                           params.batch_size)
        for i, (batch_input_seq, batch_target_seq) in enumerate(reader()):
            print(batch_input_seq.shape)
