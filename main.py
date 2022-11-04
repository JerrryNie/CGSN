# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Run Document level evidence selecting network"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import pickle
import copy
from time import sleep

sys.path.append(os.getcwd())
sys.path.append("./")
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm, trange
from glob import glob
from graph_encoder import NodePosition

from tensorboardX import SummaryWriter
from transformers import AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from graph_encoder import Config, EdgeType, NodeType, EdgePositionIndoc
from create_qasper_examples_scibert import InputFeatures, Example
from models import CGSNSciBert, CGSNLEDEncoder
from dataset import MyDataset

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

logger = logging.getLogger(__name__)

Batch = collections.namedtuple('Batch',
                               ['unique_ids', 'example_ids', 'input_ids', 'input_mask', 'segment_ids',
                                'graph_attention_mask', 'st_mask', 'st_index', 'edges_src',
                                'edges_tgt', 'edges_type', 'edges_pos', 'start_positions',
                                'end_positions', 'is_paragraph_start', 'candidate_idx',
                                'label_idx'])


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def pad_features(args, features, max_features_cnt):
    for doc_idx, doc in enumerate(features):
        while len(doc) < max_features_cnt:
            new_graph = copy.deepcopy(doc[-1].graph)
            para_start = args.max_token + args.max_sentence
            para_end = para_start + args.max_paragraph
            new_graph.st_index[para_start: para_end] = [-2] + [-1] * (para_end - para_start - 1)
            feature = InputFeatures(
                unique_id=-1,
                example_id=doc[-1].example_id,
                doc_span_index=doc[-1].doc_span_index + 1,
                tokens=None,
                token_to_orig_map=None,
                token_is_max_context=None,
                input_ids = copy.deepcopy(doc[-1].input_ids),
                input_mask=[0] * len(doc[-1].input_mask),
                segment_ids=[1] * len(doc[-1].segment_ids),
                graph_attention_mask=[False] * len(doc[-1].graph_attention_mask),
                graph=new_graph,
                is_paragraph_start=None,
                candidate_idx=[-1] * len(doc[-1].candidate_idx),
                label_idx=[-1] * len(doc[-1].label_idx) if doc[-1].label_idx else None,
                start_positions=[-2] * len(doc[-1].start_positions) if doc[-1].start_positions else None,
                end_positions=[-2] * len(doc[-1].end_positions) if doc[-1].end_positions else None
            )
            doc.append(feature)
        features[doc_idx] = doc
    return features


def batcher(args, device, is_training=False):
    def batcher_dev(features):
        if isinstance(features[0], InputFeatures):
            unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long)
            example_ids = [f.example_id for f in features]
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            candidate_idx = torch.tensor([f.candidate_idx for f in features], dtype=torch.long)
            graph_attention_mask = torch.tensor([f.graph_attention_mask for f in features], dtype=torch.bool)
            if hasattr(features[0], 'graph'):
                st_mask = torch.tensor([f.graph.st_mask for f in features], dtype=torch.long)
                st_index = torch.tensor([f.graph.st_index for f in features], dtype=torch.long)
                edges_src = [torch.tensor(f.graph.edges_src, dtype=torch.long) for f in features]
                edges_tgt = [torch.tensor(f.graph.edges_tgt, dtype=torch.long) for f in features]
                edges_type = [torch.tensor(f.graph.edges_type, dtype=torch.long) for f in features]
                edges_pos = [torch.tensor(f.graph.edges_pos, dtype=torch.long) for f in features]
            else:
                st_mask = torch.tensor([[-1] for _ in features], dtype=torch.long)
                st_index = torch.tensor([[-1] for _ in features], dtype=torch.long)
                edges_src = [torch.tensor([-1], dtype=torch.long) for _ in features]
                edges_tgt = [torch.tensor([-1], dtype=torch.long) for _ in features]
                edges_type = [torch.tensor([-1], dtype=torch.long) for _ in features]
                edges_pos = [torch.tensor([-1], dtype=torch.long) for _ in features]

            if is_training:
                if hasattr(features[0], 'graph'):
                    start_positions = torch.tensor([f.start_positions for f in features], dtype=torch.long).to(device)
                    end_positions = torch.tensor([f.end_positions for f in features], dtype=torch.long).to(device)
                else:
                    start_positions = None
                    end_positions = None
                label_idx = torch.tensor([f.label_idx for f in features], dtype=torch.long)
            else:
                start_positions = None
                end_positions = None
                label_idx = None

            for i in range(len(features)):
                edges_src[i] += st_mask.size(1) * i
                edges_tgt[i] += st_mask.size(1) * i

            edges_src = torch.cat(edges_src)
            edges_tgt = torch.cat(edges_tgt)
            edges_type = torch.cat(edges_type)
            edges_pos = torch.cat(edges_pos)

            return Batch(unique_ids=unique_ids,
                         example_ids=example_ids,
                         input_ids=input_ids.to(device),
                         input_mask=input_mask.to(device),
                         segment_ids=segment_ids.to(device),
                         st_mask=st_mask.to(device),
                         st_index=st_index,
                         candidate_idx=candidate_idx.to(device),
                         graph_attention_mask=graph_attention_mask.to(device),
                         edges_src=edges_src.to(device),
                         edges_tgt=edges_tgt.to(device),
                         edges_type=edges_type.to(device),
                         edges_pos=edges_pos.to(device),
                         start_positions=start_positions,
                         end_positions=end_positions,
                         is_paragraph_start=None,
                         label_idx=label_idx)
        elif isinstance(features[0], list):
            max_features_cnt = 0
            for feature in features:
                max_features_cnt = max(max_features_cnt, len(feature))
            if is_training:
                all_features_cnts = concat_all_gather(torch.tensor([max_features_cnt], device=device))
            else:
                all_features_cnts = torch.tensor([max_features_cnt], device=device)
            final_max_features_cnt = torch.max(all_features_cnts).item()
            features = pad_features(args, features, final_max_features_cnt)
            for feature in features:
                assert final_max_features_cnt == len(feature), '{}: {}'.format(
                    final_max_features_cnt, len(feature)
                )
            unique_ids = []
            example_ids = []
            input_ids = []
            input_mask = []
            segment_ids = []
            st_mask = []
            st_index = []
            candidate_idx = []
            graph_attention_mask = []
            edges_src = []
            edges_tgt = []
            edges_type = []
            edges_pos = []
            start_positions = []
            end_positions = []
            is_paragraph_start = []
            label_idx = []
            for doc_idx in range(final_max_features_cnt):
                new_features = []
                for feature in features:
                    new_features.append(feature[doc_idx])

                new_unique_ids = torch.tensor([f.unique_id for f in new_features], dtype=torch.long)
                new_example_ids = [f.example_id for f in new_features]
                new_input_ids = torch.tensor([f.input_ids for f in new_features], dtype=torch.long)
                new_input_mask = torch.tensor([f.input_mask for f in new_features], dtype=torch.long)
                new_segment_ids = torch.tensor([f.segment_ids for f in new_features], dtype=torch.long)
                new_candidate_idx = torch.tensor([f.candidate_idx for f in new_features], dtype=torch.long)
                new_graph_attention_mask = torch.tensor([f.graph_attention_mask for f in new_features], dtype=torch.long)
                if hasattr(new_features[0], 'graph'):
                    new_st_mask = torch.tensor([f.graph.st_mask for f in new_features], dtype=torch.long)
                    new_st_index = torch.tensor([f.graph.st_index for f in new_features], dtype=torch.long)
                    new_edges_src = [torch.tensor(f.graph.edges_src, dtype=torch.long) for f in new_features]
                    new_edges_tgt = [torch.tensor(f.graph.edges_tgt, dtype=torch.long) for f in new_features]
                    new_edges_type = [torch.tensor(f.graph.edges_type, dtype=torch.long) for f in new_features]
                    new_edges_pos = [torch.tensor(f.graph.edges_pos, dtype=torch.long) for f in new_features]
                else:
                    new_st_mask = torch.tensor([[-1] for _ in new_features], dtype=torch.long)
                    new_st_index = torch.tensor([[-1] for _ in new_features], dtype=torch.long)
                    new_edges_src = [torch.tensor([-1], dtype=torch.long) for _ in new_features]
                    new_edges_tgt = [torch.tensor([-1], dtype=torch.long) for _ in new_features]
                    new_edges_type = [torch.tensor([-1], dtype=torch.long) for _ in new_features]
                    new_edges_pos = [torch.tensor([-1], dtype=torch.long) for _ in new_features]        

                if is_training:
                    if hasattr(new_features[0], 'graph'):
                        new_start_positions = torch.tensor([f.start_positions for f in new_features], dtype=torch.long).to(device)
                        new_end_positions = torch.tensor([f.end_positions for f in new_features], dtype=torch.long).to(device)
                    else:
                        new_start_positions = None
                        new_end_positions = None
                    new_label_idx = None
                else:
                    new_start_positions = None
                    new_end_positions = None
                    new_label_idx = None

                assert len(new_features) == 1
                for i in range(len(new_features)):
                    new_edges_src[i] += new_st_mask.size(1) * i
                    new_edges_tgt[i] += new_st_mask.size(1) * i

                new_edges_src = torch.cat(new_edges_src)
                new_edges_tgt = torch.cat(new_edges_tgt)
                new_edges_type = torch.cat(new_edges_type)
                new_edges_pos = torch.cat(new_edges_pos)

                unique_ids.append(new_unique_ids)
                example_ids.append(new_example_ids)
                input_ids.append(new_input_ids)
                input_mask.append(new_input_mask)
                segment_ids.append(new_segment_ids)
                st_mask.append(new_st_mask)
                st_index.append(new_st_index)
                candidate_idx.append(new_candidate_idx)
                graph_attention_mask.append(new_graph_attention_mask)
                edges_src.append(new_edges_src)
                edges_tgt.append(new_edges_tgt)
                edges_type.append(new_edges_type)
                edges_pos.append(new_edges_pos)
                start_positions.append(new_start_positions)
                end_positions.append(new_end_positions)
                is_paragraph_start.append(None)
                label_idx.append(new_label_idx)

            return Batch(unique_ids=unique_ids,
                         example_ids=example_ids,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         st_mask=st_mask,
                         st_index=st_index,
                         candidate_idx=candidate_idx,
                         graph_attention_mask=graph_attention_mask,
                         edges_src=edges_src,
                         edges_tgt=edges_tgt,
                         edges_type=edges_type,
                         edges_pos=edges_pos,
                         start_positions=start_positions,
                         end_positions=end_positions,
                         is_paragraph_start=is_paragraph_start,
                         label_idx=label_idx)

    return batcher_dev


def eval_model(args, device, model, data_pattern, prefix='', worker_init_fn=None):
    for data_path in glob(data_pattern):
        eval_dataset = MyDataset(args, data_path, is_training=False)
        eval_examples = eval_dataset.examples
        eval_features = eval_dataset.features

        args.predict_batch_size = args.predict_batch_size * max(1, args.n_gpu)
        logger.info("***** Running predictions {} *****".format(prefix))
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        eval_sampler = SequentialSampler(eval_features)
        eval_dataloader = DataLoader(eval_features, sampler=eval_sampler, batch_size=args.predict_batch_size,
                                     collate_fn=batcher(args, device, is_training=False), num_workers=0,
                                     worker_init_fn=worker_init_fn)
        model.eval()
        logger.info("Start evaluating")
        qid_to_logits = {}
        all_qids = []
        for batch in tqdm(eval_dataloader, desc='eval'):
            seq_len = len(batch.input_ids)
            history_meta_hiddens = [None]
            evidence_meta_hiddens = [None]
            for seq_pos in range(seq_len):
                with torch.no_grad():
                    outputs = model(input_ids=batch.input_ids[seq_pos].to(device),
                                    attention_mask=batch.input_mask[seq_pos].to(device),
                                    token_type_ids=batch.segment_ids[seq_pos].to(device),
                                    st_mask=batch.st_mask[seq_pos].to(device),
                                    st_index=batch.st_index[seq_pos],
                                    graph_attention_mask=batch.graph_attention_mask[seq_pos].to(device),
                                    meta_hiddens=history_meta_hiddens[-1],
                                    evidence_hiddens=evidence_meta_hiddens[-1],
                                    edges=(batch.edges_src[seq_pos].to(device), batch.edges_tgt[seq_pos].to(device),
                                           batch.edges_type[seq_pos].to(device), batch.edges_pos[seq_pos].to(device)),
                                    is_paragraph_start=None,
                                    candidate_idx=batch.candidate_idx[seq_pos].to(device))
                    batch_para_logits, batch_para_ref_indexes = outputs[0], outputs[1]
                    meta_hiddens = outputs[2]
                    evidence_hiddens = outputs[3]
                    history_meta_hiddens.append(meta_hiddens.cuda())
                    evidence_meta_hiddens.append(evidence_hiddens.cuda())
                for i, unique_id in enumerate(batch.unique_ids[seq_pos]):
                    para_logits = F.softmax(batch_para_logits[i], dim=-1)[:, 1]
                    para_logits = para_logits.detach().cpu().tolist()
                    para_ref_indexes = batch_para_ref_indexes[i].detach().cpu().tolist()
                    assert len(para_logits) == len(para_ref_indexes)
                    unique_id = int(unique_id)
                    example_id = batch.example_ids[seq_pos][i]
                    if example_id not in qid_to_logits:
                        la_candidates = None
                        for example in eval_examples:
                            if example.example_id == example_id:
                                la_candidates = example.la_candidates
                                break
                        qid_to_logits[example_id] = [[] for _ in la_candidates]
                        all_qids.append(example_id)
                    for para_ref_index, para_logit in zip(para_ref_indexes, para_logits):
                        if para_ref_index >= 0:
                            qid_to_logits[example_id][para_ref_index].append(para_logit)

    assert len(all_qids) == len(qid_to_logits)
    all_results = []
    for query_id, qid in enumerate(all_qids):
        for para_id, logits in enumerate(qid_to_logits[qid]):
            label = sum(logits) / len(logits) if len(logits) > 0 else 0
            all_results.append(
                f'{query_id}\t{para_id}\t{label}\n'
            )
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.txt".format(prefix))
    with open(output_prediction_file, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(line)
    print(f'[!] write the rest into the file {output_prediction_file}')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="bert-base-uncased")
    parser.add_argument("--my_config", default=None, type=str, required=True)
    parser.add_argument("--feature_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--train_pattern", default=None, type=str, help="training data path.")
    parser.add_argument("--test_pattern", default=None, type=str, help="test data path.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--max_para_len', default=256, type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save_epochs', type=int, default=None)
    parser.add_argument('--max_epochs', default=None, type=int, help='early stop after max_epochs')
    parser.add_argument('--num_hidden_layers', default=None, type=int)
    parser.add_argument('--meta_gat_hops', type=int, default=None,
                        help='the number of hops among meta nodes')
    parser.add_argument('--requires_grad', default='all', type=str)
    parser.add_argument('--neighborhops', type=int, default=1, help='the number of hops between neighbor nodes')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--led_encoder', action='store_true')
    parser.add_argument('--memory_hops', type=int, default=1)
    parser.add_argument('--indoc_num', default=16, type=int)
    parser.add_argument('--meta_sentence_num', default=None, type=int)
    parser.add_argument('--meta_paragraph_num', default=None, type=int)
    parser.add_argument('--meta_doc_num', default=None, type=int)
    args = parser.parse_args()
    print(args)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    args.n_gpu = n_gpu

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_pattern:
            raise ValueError(
                "If `do_train` is True, then `train_pattern` must be specified.")

    if args.do_predict:
        if not args.test_pattern:
            raise ValueError(
                "If `do_predict` is True, then `test_pattern` must be specified.")
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if args.local_rank in [-1, 0]:
            torch.distributed.barrier()

    # Prepare model
    my_config = Config(args.my_config)
    my_config.num_edge_types = sum(EdgePositionIndoc.max_edge_types)
    my_config.forward_edges = [EdgeType.TOKEN_TO_SENTENCE,
                               EdgeType.SENTENCE_TO_PARAGRAPH,
                               EdgeType.PARAGRAPH_TO_DOCUMENT]
    print(my_config)
    my_config.meta_sentence_num = args.meta_sentence_num
    my_config.meta_paragraph_num = args.meta_paragraph_num
    my_config.meta_doc_num = args.meta_doc_num
    my_config.neighborhops = args.neighborhops
    my_config.dropout = args.dropout
    my_config.memory_hops = args.memory_hops
    my_config.meta_gat_hops = args.meta_gat_hops
    my_config.requires_grad = args.requires_grad
    args.max_seq_length = my_config.max_indoc_token_len
    args.max_indoc_token = my_config.max_indoc_token_len
    args.max_indoc_sentence = my_config.max_indoc_sentence_len
    args.max_indoc_paragraph = my_config.max_indoc_paragraph_len
    my_config.indoc_num = my_config.indoc_num if args.indoc_num is None else args.indoc_num
    args.indoc_num = my_config.indoc_num if args.indoc_num is None else args.indoc_num
    args.max_token = args.max_indoc_token * args.indoc_num
    args.max_sentence = args.max_indoc_sentence * args.indoc_num
    args.max_paragraph = args.max_indoc_paragraph * args.indoc_num
    if args.num_hidden_layers is not None:
        my_config.num_hidden_layers = args.num_hidden_layers
    bert_config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.led_encoder:
        model = CGSNLEDEncoder(model_name_or_path=args.model_name_or_path,
                               bert_config=bert_config, my_config=my_config)
    else:
        model = CGSNSciBert(model_name_or_path=args.model_name_or_path,
                            bert_config=bert_config, my_config=my_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    num_train_features = None
    num_train_optimization_steps = None
    if args.do_train:
        num_train_features = 0
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        for data_path in glob(args.train_pattern):
            train_dataset = MyDataset(args, data_path, is_training=True)
            num_train_features += len(train_dataset.features)
        if args.local_rank == 0:
            torch.distributed.barrier()

        num_train_optimization_steps = int(
            num_train_features / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if args.warmup_steps > 0:
            args.warmup_proportion = min(args.warmup_proportion, args.warmup_steps / num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                                                    num_training_steps=num_train_optimization_steps)
    global_step = 0
    report_step = 0
    seq_step = 0
    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
            logger.info("***** Running training *****")
            logger.info("  Num split examples = %d", num_train_features)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        tr_loss = 0.0
        nb_tr_examples = 0
        logging.info('Total training files: [{}]'.format(len(glob(args.train_pattern))))
        for epoch_num in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            for data_idx, data_path in enumerate(sorted(glob(args.train_pattern))):
                logging.info("[{}]: Reading data from {}.".format(data_idx, data_path))
                train_dataset = MyDataset(args, data_path, is_training=True)
                train_features = train_dataset.features
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_features)
                else:
                    train_sampler = DistributedSampler(train_features)
                train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size,
                                              collate_fn=batcher(args, device, is_training=True), num_workers=0,
                                              worker_init_fn=_init_fn)

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                    seq_step += 1
                    seq_len = len(batch.input_ids)
                    history_meta_hiddens = [None]
                    evidence_meta_hiddens = [None]
                    for seq_pos in range(seq_len):
                        unique_ids = batch.unique_ids[seq_pos].tolist()
                        valid_cnt = len(unique_ids) - unique_ids.count(-1)
                        total = len(unique_ids)
                        outputs = model(input_ids=batch.input_ids[seq_pos].to(device),
                                        attention_mask=batch.input_mask[seq_pos].to(device),
                                        token_type_ids=batch.segment_ids[seq_pos].to(device),
                                        st_mask=batch.st_mask[seq_pos].to(device),
                                        st_index=batch.st_index[seq_pos].to(device),
                                        graph_attention_mask=batch.graph_attention_mask[seq_pos].to(device),
                                        meta_hiddens=history_meta_hiddens[-1],
                                        evidence_hiddens=evidence_meta_hiddens[-1],
                                        edges=(batch.edges_src[seq_pos].to(device),
                                               batch.edges_tgt[seq_pos].to(device),
                                               batch.edges_type[seq_pos].to(device),
                                               batch.edges_pos[seq_pos].to(device)),
                                        is_paragraph_start=None,
                                        candidate_idx=batch.candidate_idx[seq_pos].to(device),
                                        label_idx=None,
                                        start_positions=batch.start_positions[seq_pos])
                        loss = outputs[0]
                        loss *= (float(valid_cnt) / total)
                        meta_hiddens = outputs[3]
                        evidence_hiddens = outputs[4]
                        history_meta_hiddens.append(meta_hiddens.cuda())
                        evidence_meta_hiddens.append(evidence_hiddens.cuda())
                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.local_rank != -1:
                            pass
                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()
                            global_step += 1
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if global_step % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            report_step += 1
                            if args.local_rank in [-1, 0]:
                                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], report_step)
                                record_loss = loss.item() * args.gradient_accumulation_steps
                                if valid_cnt > 0:
                                    record_loss /= (float(valid_cnt) / total)
                                tb_writer.add_scalar('loss', record_loss, report_step)
                                tb_writer.add_scalar('gradient_norm', grad_norm, report_step)
                                tr_loss += loss.item()
                                nb_tr_examples += 1
                    if global_step % args.gradient_accumulation_steps == 0:
                        scheduler.step()
                if data_idx + 1 < len(glob(args.train_pattern)) and args.local_rank in [-1, 0] and args.save_epochs and (epoch_num + 1) % args.save_epochs == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'part-{}-epoch-{}'.format(data_idx, epoch_num))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
                    logger.info("Saving model epoch to %s", output_dir)
            if args.local_rank in [-1, 0] and args.save_epochs and (epoch_num + 1) % args.save_epochs == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'epoch-{}'.format(epoch_num))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
                logger.info("Saving model epoch to %s", output_dir)
            if args.max_epochs and epoch_num == args.max_epochs:
                break
        if args.local_rank in [-1, 0]:
            tb_writer.close()

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
        if args.led_encoder:
            model = CGSNLEDEncoder(model_name_or_path=args.model_name_or_path,
                                                   bert_config=bert_config, my_config=my_config)
        else:
            model = CGSNSciBert(model_name_or_path=args.model_name_or_path,
                                         bert_config=bert_config, my_config=my_config)
        model.load_state_dict(torch.load(output_model_file))
        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        checkpoints = list(os.path.dirname(c)
                           for c in sorted(glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                                                recursive=True)))
        p_checkpoints = []
        if len(checkpoints) > 1:
            final_model = args.output_dir
            tmp_ckpts = []
            for ckpt in checkpoints:
                tail = ckpt.split('-')[-1]
                if tail.isdigit():
                    tmp_ckpts.append(ckpt)
            checkpoints = tmp_ckpts
            print(len(checkpoints))
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
            p_checkpoints = list(filter(lambda x: 'part-' in os.path.split(x)[-1], checkpoints))
            checkpoints = list(filter(lambda x: 'part-' not in os.path.split(x)[-1], checkpoints))
            if len(checkpoints) == 0 and os.path.exists(os.path.join(final_model, 'pytorch_model.bin')):
                checkpoints.append(final_model)
                
        logger.info("Evaluate the following epoch: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            ckpt_file = os.path.join(checkpoint, WEIGHTS_NAME)
            state_dict = torch.load(ckpt_file)
            if (args.local_rank != -1 or n_gpu > 1) and hasattr(model, 'module') and not hasattr(state_dict, 'module'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k
                    new_state_dict[name] = v
            else:
                new_state_dict = state_dict
            model.load_state_dict(new_state_dict)
            model.to(device)

            if global_step.isdigit():
                global_step = args.prefix + global_step
            else:
                global_step = args.prefix
            eval_model(args, device, model, args.test_pattern, prefix=global_step, worker_init_fn=_init_fn)

        logger.info("Evaluate the following p_epoch: %s", p_checkpoints)
        for p_checkpoint in p_checkpoints:
            global_step = 'p_' + p_checkpoint.split('-')[-3] + 'e_' + p_checkpoint.split('-')[-1]
            ckpt_file = os.path.join(p_checkpoint, WEIGHTS_NAME)
            state_dict = torch.load(ckpt_file)
            if (args.local_rank != -1 or n_gpu > 1) and hasattr(model, 'module') and not hasattr(state_dict, 'module'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k
                    new_state_dict[name] = v
            else:
                new_state_dict = state_dict
            model.load_state_dict(new_state_dict)
            model.to(device)

            global_step = args.prefix + global_step
            eval_model(args, device, model, args.test_pattern, prefix=global_step, worker_init_fn=_init_fn)
            

if __name__ == "__main__":
    main()
