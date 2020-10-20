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
"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import tarfile
import urllib

import six
import oneflow.core.record.record_pb2 as ofrecord
import struct

from oneflow_transformer.utils import tokenizer

# Data sources for training/evaluating the transformer translation model.
# If any of the training sources are changed, then either:
#   1) use the flag `--search` to find the best min count or
#   2) update the _TRAIN_DATA_MIN_COUNT constant.
# min_count is the minimum number of times a token must appear in the data
# before it is added to the vocabulary. "Best min count" refers to the value
# that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.
_TRAIN_DATA_SOURCES = [
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
# Use pre-defined minimum count to generate subtoken vocabulary.
_TRAIN_DATA_MIN_COUNT = 6

_EVAL_DATA_SOURCES = [
    {
        "url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        "input": "newstest2013.en",
        "target": "newstest2013.de",
    }
]

# Vocabulary constants
_TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
_TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
VOCAB_FILE = "vocab.ende.%d" % _TARGET_VOCAB_SIZE

# Strings to inclue in the generated files.
_PREFIX = "wmt32k"
_ENCODE_TAG = "encoded"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.

# Number of files to split train and evaluation data
_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1


def find_file(path, filename, max_depth=5):
    """Returns full filepath if the file is in path or a subdirectory."""
    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)

        # Don't search past max_depth
        depth = root[len(path) + 1:].count(os.sep)
        if depth > max_depth:
            del dirs[:]  # Clear dirs
    return None


###############################################################################
# Download and extraction functions
###############################################################################
def get_raw_files(raw_dir, data_source):
    """Return raw files from source. Downloads/extracts if needed.

  Args:
    raw_dir: string directory to store raw files
    data_source: dictionary with
      {"url": url of compressed dataset containing input and target files
       "input": file with data in input language
       "target": file with data in target language}

  Returns:
    dictionary with
      {"inputs": list of files containing data in input language
       "targets": list of files containing corresponding data in target language
      }
  """
    raw_files = {
        "inputs": [],
        "targets": [],
    }  # keys
    for d in data_source:
        input_file, target_file = download_and_extract(
            raw_dir, d["url"], d["input"], d["target"])
        raw_files["inputs"].append(input_file)
        raw_files["targets"].append(target_file)
    return raw_files


def download_report_hook(count, block_size, total_size):
    """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
    percent = int(count * block_size * 100 / total_size)
    print("\r%d%%" % percent + " completed", end="\r")


def download_from_url(path, url):
    """Download content from a url.

  Args:
    path: string directory where file will be downloaded
    url: string url

  Returns:
    Full path to downloaded file
  """
    filename = url.split("/")[-1]
    found_file = find_file(path, filename, max_depth=0)
    if found_file is None:
        filename = os.path.join(path, filename)
        print("Downloading from %s to %s." % (url, filename))

        inprogress_filepath = filename + ".incomplete"
        inprogress_filepath, _ = urllib.urlretrieve(
            url, inprogress_filepath, reporthook=download_report_hook)
        # Print newline to clear the carriage return from the download progress.
        print()
        os.rename(inprogress_filepath, filename)
        return filename
    else:
        print("Already downloaded: %s (at %s)." % (url, found_file))
        return found_file


def download_and_extract(path, url, input_filename, target_filename):
    """Extract files from downloaded compressed archive file.

  Args:
    path: string directory where the files will be downloaded
    url: url containing the compressed input and target files
    input_filename: name of file containing data in source language
    target_filename: name of file containing data in target language

  Returns:
    Full paths to extracted input and target files.

  Raises:
    OSError: if the the download/extraction fails.
  """
    # Check if extracted files already exist in path
    input_file = find_file(path, input_filename)
    target_file = find_file(path, target_filename)
    if input_file and target_file:
        print("Already downloaded and extracted %s." % url)
        return input_file, target_file

    # Download archive file if it doesn't already exist.
    compressed_file = download_from_url(path, url)

    # Extract compressed files
    print("Extracting %s." % compressed_file)
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(path)

    # Return filepaths of the requested files.
    input_file = find_file(path, input_filename)
    target_file = find_file(path, target_filename)

    if input_file and target_file:
        return input_file, target_file

    raise OSError("Download/extraction failed for url %s to path %s" %
                  (url, path))


def txt_line_iterator(path):
    """Iterate through lines of file."""
    with open(path) as f:
        for line in f:
            yield line.strip()


def compile_files(raw_dir, raw_files, tag):
    """Compile raw files into a single file for each language.

  Args:
    raw_dir: Directory containing downloaded raw files.
    raw_files: Dict containing filenames of input and target data.
      {"inputs": list of files containing data in input language
       "targets": list of files containing corresponding data in target language
      }
    tag: String to append to the compiled filename.

  Returns:
    Full path of compiled input and target files.
  """
    print("Compiling files with tag %s." % tag)

    filename = "%s-%s" % (_PREFIX, tag)
    input_compiled_file = os.path.join(raw_dir, filename + ".lang1")
    target_compiled_file = os.path.join(raw_dir, filename + ".lang2")

    with open(input_compiled_file, mode="w") as input_writer:
        with open(target_compiled_file, mode="w") as target_writer:
            for i in range(len(raw_files["inputs"])):
                input_file = raw_files["inputs"][i]
                target_file = raw_files["targets"][i]

                print("Reading files %s and %s." % (input_file, target_file))

                write_file(input_writer, input_file)
                write_file(target_writer, target_file)
    return input_compiled_file, target_compiled_file


def write_file(writer, filename):
    """Write all of lines from file using the writer."""
    for line in txt_line_iterator(filename):
        writer.write(line)
        writer.write("\n")


###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(
        subtokenizer, data_dir, raw_files, tag, total_shards):
    """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
    # Create a file for each shard.
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
                 for n in range(total_shards)]

    if all_exist(filepaths):
        print("Files with tag %s already exist." % tag)
        return filepaths

    print("Saving files with tag %s." % tag)
    input_file = raw_files[0]
    target_file = raw_files[1]

    # Write examples to each shard in round robin order.
    tmp_filepaths = [fname + ".incomplete" for fname in filepaths]

    writers = [open(fname, mode="wb") for fname in tmp_filepaths]

    counter, shard = 0, 0
    for counter, (input_line, target_line) in enumerate(zip(
            txt_line_iterator(input_file), txt_line_iterator(target_file))):
        if counter > 0 and counter % 100000 == 0:
            print("\t Saving case %d." % counter)

        serilized_bytes, features_length = dict_to_example(
            {"inputs": subtokenizer.encode(input_line, add_eos=True),
             "targets": subtokenizer.encode(target_line, add_eos=True)})

        # Pack length
        length = struct.pack("q", features_length)

        # Write length and serilized Bytes Feature
        writers[shard].write(length)
        writers[shard].write(serilized_bytes)

        shard = (shard + 1) % total_shards

    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        os.rename(tmp_name, final_name)

    print("Saved %d Examples", counter)

    return filepaths


def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%s-%.5d-of-%.5d" % (_PREFIX, _ENCODE_TAG, tag, shard_num, total_shards))


def shuffle_records(fname, dataset_dir, data_part_num):
    """Shuffle records in a single file."""
    print("Shuffling records in file %s" % fname)

    tmp_fname = fname + ".unshuffled"

    os.rename(fname, tmp_fname)  # Rename the file name as suffix with ".unshuffled"

    records = []

    with open(tmp_fname, "rb") as f:
        while True:
            try:
                length = struct.unpack("q", f.read(8))
                serilizedBytes = f.read(length[0])
                ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
                records.append([length[0], ofrecord_features])
                if len(records) % 100000 == 0:
                    print("\tRead: %d", len(records))
            except:
                break

    random.shuffle(records)

    # Write shuffled records to original file name

    final_filename = os.path.join(dataset_dir, "part-{}".format(data_part_num))

    with open(final_filename, 'wb') as f:
        for count, record in enumerate(records):
            length = record[0]  # Get the length
            feature = record[1]  # Get the feature
            feature = feature.SerializeToString()  # Serialize it
            f.write(struct.pack('q', length))  # Write Length
            f.write(feature)  # Write Feature

    os.remove(tmp_fname)


def OneFlow_int64_feature(value):
    """Generate OFRecord int64 Feature

    Args:
        value ([type]): Input 1D int Array

    Returns:
        [type]: OFRecord int64 Feature
    """
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def dict_to_example(dictionary):
    """
    Pack and Serilize Feature
    :param dictionary: The input Dictionary
    :return: The Serilized Feature and the length of Feature
    """
    feature_pack = {}
    for k, v in six.iteritems(dictionary):
        # Like: {'inputs': int64_List{xxxxxx}, 'targets': int64_List{xxx}}
        feature_pack[k] = OneFlow_int64_feature(value=v)

    # convert to ofrecord_features
    ofrecord_features = ofrecord.OFRecord(feature=feature_pack)
    # Serilized Features
    serialized_bytes = ofrecord_features.SerializeToString()

    # Compute Length
    features_length = ofrecord_features.ByteSize()

    return serialized_bytes, features_length


def all_exist(filepaths):
    """Returns true if all files in the list exist."""
    for fname in filepaths:
        if not os.path.exists(fname):
            return False
    return True


def make_dir(path):
    if not os.path.exists(path):
        print("Creating directory %s" % path)
        os.makedirs(path)


def main(unused_argv):
    """Obtain training and evaluation data for the Transformer model."""

    # Make directory for raw_dir
    make_dir(FLAGS.raw_dir)
    # Make directory for data_dir
    make_dir(FLAGS.data_dir)
    # Make directory for OFRecord_dir, Train and Eval
    make_dir(os.path.join(FLAGS.OFRecord_dir, "train"))
    make_dir(os.path.join(FLAGS.OFRecord_dir, "eval"))

    # Get paths of download/extracted training and evaluation files.

    # tf.logging.info("Step 1/4: Downloading data from source")
    print("Step 1/4: Downloading data from source")
    train_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
    eval_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)

    # Create subtokenizer based on the training files.

    # tf.logging.info("Step 2/4: Creating subtokenizer and building vocabulary")
    print("Step 2/4: Creating subtokenizer and building vocabulary")
    train_files_flat = train_files["inputs"] + train_files["targets"]
    vocab_file = os.path.join(FLAGS.data_dir, VOCAB_FILE)
    subtokenizer = tokenizer.Subtokenizer.init_from_files(
        vocab_file, train_files_flat, _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
        min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT)
    #
    # # tf.logging.info("Step 3/4: Compiling training and evaluation data")
    print("Step 3/4: Compiling training and evaluation data")
    compiled_train_files = compile_files(FLAGS.raw_dir, train_files, _TRAIN_TAG)
    compiled_eval_files = compile_files(FLAGS.raw_dir, eval_files, _EVAL_TAG)

    # # Tokenize and save data as Examples in the TFRecord format.
    # # tf.logging.info("Step 4/4: Preprocessing and saving data")
    print("Step 4/4: Preprocessing and saving data")

    train_ofrecord_files = encode_and_save_files(
        subtokenizer, FLAGS.data_dir, compiled_train_files, _TRAIN_TAG,
        _TRAIN_SHARDS)

    # TODO: In tensorflow mlperf, it doesn't shuffle the eval file
    # I just want to split to the dataset/eval directory, so i use shuffle in eval
    eval_ofrecord_files = encode_and_save_files(
        subtokenizer, FLAGS.data_dir, compiled_eval_files, _EVAL_TAG,
        _EVAL_SHARDS)

    for num, fname in enumerate(train_ofrecord_files):
        shuffle_records(fname, os.path.join(FLAGS.OFRecord_dir, "train"), num)

    for num, fname in enumerate(eval_ofrecord_files):
        shuffle_records(fname, os.path.join(FLAGS.OFRecord_dir, "eval"), num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-dd", type=str, default="./translate_ende",
        help="[default: %(default)s] Directory for where the "
             "translate_ende_wmt32k dataset is saved.",
        metavar="<DD>")
    parser.add_argument(
        "--raw_dir", "-rd", type=str, default="./raw_data",
        help="[default: %(default)s] Path where the raw data will be downloaded "
             "and extracted.",
        metavar="<RD>")
    parser.add_argument(
        "--OFRecord_dir", "-od", type=str, default="./dataset",
        help="[default: %(default)s] The Path of OFRecord dataset",
        metavar="<OD>")
    parser.add_argument(
        "--search", action="store_true",
        help="If set, use binary search to find the vocabulary set with size"
             "closest to the target size (%d)." % _TARGET_VOCAB_SIZE)

    FLAGS, unparsed = parser.parse_known_args()
    main(sys.argv)
