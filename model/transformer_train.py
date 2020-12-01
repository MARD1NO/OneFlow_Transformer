from oneflow_transformer.model.generate_mask import create_masks
from oneflow_transformer.model.transformer import Transformer
from oneflow_transformer.utils.reader import read_data
from oneflow_transformer.model.model_params import PathParams, TrainParams
from oneflow_transformer.utils import tokenizer_v3
from oneflow_transformer.model.model_utils import loss_function
import oneflow as flow
import oneflow.typing as tp

# Config the Params for FilePath and Train params
params = PathParams()
train_params = TrainParams()


@flow.global_function(type="train")
def transformer_train_job(input: tp.Numpy.Placeholder(shape=(params.batch_size, params.max_length), dtype=flow.int64),
                          target: tp.Numpy.Placeholder(shape=(params.batch_size, params.max_length),
                                                       dtype=flow.int64)) -> tp.Numpy:
    """
    The transformer training Job
    :param input: The input Sequence, we fix the shape to (_batch_size, _max_length)
    :param target: The target Sequence, we fix the shape to (_batch_size, _max_length)
    :return: Return the loss value.
    """
    sample_transformer = Transformer(
        num_layers=6, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=params.TARGET_VOCAB_SIZE, target_vocab_size=params.TARGET_VOCAB_SIZE,
        pe_input=params.TARGET_VOCAB_SIZE, pe_target=params.TARGET_VOCAB_SIZE)

    tar_inp = flow.slice(target, begin=[None, 1], size=[None, params.max_length - 1])  # (batch, seq_len - 1)
    tar_real = flow.slice(target, begin=[None, 0], size=[None, params.max_length - 1])  # (batch, seq_len - 1)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, tar_inp)

    prediction, _ = sample_transformer(input,
                                       tar_inp,
                                       training=False,
                                       enc_padding_mask=enc_padding_mask,
                                       look_ahead_mask=combined_mask,
                                       dec_padding_mask=dec_padding_mask)

    loss = loss_function(tar_real, prediction)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)

    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()
    train_file_num = len(params.raw_file_input_dir)
    subtokenizer = tokenizer_v3.Subtokenizer.init_from_files(
        params.vocab_dir, params.raw_file_dir, params.TARGET_VOCAB_SIZE, params.TARGET_THRESHOLD,
        min_count=None)

    for epoch_id in range(train_params.epoch):
        loss = 0
        for train_file_id in range(train_file_num):
            reader = read_data(params.raw_file_input_dir[train_file_id],
                               params.raw_file_target_dir[train_file_id],
                               subtokenizer,
                               params.max_length,
                               params.batch_size)
            for i, (batch_input_seq, batch_target_seq) in enumerate(reader()):
                # print(batch_input_seq.shape)
                _loss = transformer_train_job(batch_input_seq, batch_target_seq)
                loss += _loss
                # print("Epoch: {}, Iter: {}, Transformer Loss is: {}".format(epoch_id, i, _loss))
        print("Epoch: {}, Total Loss is: {}".format(epoch_id, loss))
