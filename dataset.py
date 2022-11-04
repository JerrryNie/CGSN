import os
import sys
import six
import json
import gzip
import logging
import copy
from create_qasper_examples_scibert import Example, convert_qasper

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDataset(object):
    def __init__(self, args, input_file, is_training):
        if not is_training:
            self.examples = self.read_examples(input_path=input_file)

        prefix = "cached_{0}_{1}_{2}_{3}_indoc{4}".format(list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                                          str(args.max_seq_length),
                                                          str(args.doc_stride),
                                                          str(args.max_query_length),
                                                          str(args.indoc_num))
        if args.led_encoder:
            prefix += '_led'
        prefix += '_train' if is_training else '_test'
        prefix = os.path.join(args.feature_path, prefix)

        cached_path = os.path.join(prefix, os.path.split(input_file)[1] + ".pkl")
        self.features = self.read_features(cached_path, is_training)

    @staticmethod
    def read_examples(input_path):
        logging.info("Reading examples from {}.".format(input_path))
        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            logging.info("Reading data from {}.".format(input_path))
            input_data = json.load(f)
            qpapers = convert_qasper(input_data)
            for json_example in qpapers:
                example_id = json_example["example_id"]
                la_candidates = json_example["long_answer_candidates"]

                examples.append(
                    Example(example_id=example_id,
                              question_tokens=None,
                              doc_tokens=None,
                              la_candidates=la_candidates,
                              annotation=None))
        return examples

    @staticmethod
    def read_features(cached_path, is_training=False):
        if not os.path.exists(cached_path):
            logging.info("{} doesn't exists.".format(cached_path))
            exit(0)
        logging.info("Reading docs from {}.".format(cached_path))
        with open(cached_path, "rb") as reader:
            docs = pickle.load(reader)

        unique_id = 0
        for doc in docs:
            for i, feature in enumerate(doc):
                feature.unique_id = unique_id
                unique_id += 1
        return docs
