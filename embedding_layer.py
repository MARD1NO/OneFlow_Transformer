"""
author: zzk

Write Embedding Layer
"""

import oneflow as flow
import oneflow.typing as tp


def EmbeddingLayer(input_ids_blob,
                   vocab_size,
                   embedding_size=128,
                   initializer_range=0.02,
                   word_embedding_name="Embedding_Layer"):
    """
    Embedding Layer
    :param input_ids_blob:The input ID Blob
    :param vocab_size: The input Vocab size
    :param embedding_size: The embedding Size
    :param initializer_range: The range of Initializer, Use flow.truncated_normal
    :param word_embedding_name: The name of Embedding variable
    :return: The output and the Embedding table.
    """
    embedding_table = flow.get_variable(name=word_embedding_name+"_Embed",
                                        shape=[vocab_size, embedding_size],
                                        dtype=flow.float32,
                                        initializer=flow.truncated_normal(initializer_range))
    output = flow.gather(params=embedding_table, indices=input_ids_blob, axis=0)
    return output


# # Test
# @flow.global_function()
# def test_Embedding(x: tp.Numpy.Placeholder(shape=(1, 4), dtype=flow.int32)) -> tp.Numpy:
#     out = EmbeddingLayer(x, 10, 3)
#     return out
#
#
# check = flow.train.CheckPoint()
# check.init()
#
# import numpy as np
#
# x = np.array([[0, 2, 0, 5]]).astype(np.int32)
# out = test_Embedding(x)
# print(out)
