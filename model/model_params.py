import os


class PathParams(object):
    def __init__(self):
        self.TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
        self.TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
        self.VOCAB_FILE = "vocab.ende.%d" % self.TARGET_VOCAB_SIZE

        self.train_data_dir = "/home/zzk/Code/oneflow_transformer/model/raw_data_mini/"

        self.TRAIN_DATA_SOURCES = [
            {
                "url": "http://data.statmt.org/wmt17/translation-task/"
                       "training-parallel-nc-v12.tgz",
                "input": "news-commentary-v12.de-en.en",
                "target": "news-commentary-v12.de-en.de",
            },
            {
                "url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                "input": "commoncrawl.de-en.en",
                "target": "commoncrawl.de-en.de",
            },
            {
                "url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                "input": "europarl-v7.de-en.en",
                "target": "europarl-v7.de-en.de",
            },
        ]

        self.raw_file_input_dir = [os.path.join(self.train_data_dir, _data["input"])
                                   for _data in self.TRAIN_DATA_SOURCES]

        self.raw_file_target_dir = [os.path.join(self.train_data_dir, _data["target"])
                                    for _data in self.TRAIN_DATA_SOURCES]

        self.raw_file_dir = self.raw_file_input_dir + self.raw_file_target_dir

        self.data_dir = "/home/zzk/Code/oneflow_transformer/model/translate_ende/"
        self.vocab_dir = os.path.join(self.data_dir, "vocab.ende.32768")

        self.max_length = 256
        self.batch_size = 18


class TrainParams(object):
    def __init__(self):
        self.epoch = 100
        self.lr = 0.01
        self.opt = "adam"
