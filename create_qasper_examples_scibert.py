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
"""Construct <question, paragraph> pairs as the features."""
from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import enum
import os
import random
import sys
import copy
import nltk
from io import open
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


from spacy.lang.en import English
from graph_encoder import NodePosition, GraphIndoc, EdgeType, get_edge_position_indoc, Config

nlp = English()
nlp.add_pipe("sentencizer")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 question_tokens,
                 doc_tokens,
                 la_candidates,
                 annotation):
        self.example_id = example_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.la_candidates = la_candidates
        self.annotation = annotation


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 graph,
                 graph_attention_mask,
                 is_paragraph_start,
                 candidate_idx,
                 label_idx=None,
                 start_positions=None,
                 end_positions=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.graph = graph
        self.graph_attention_mask = graph_attention_mask
        self.is_paragraph_start = is_paragraph_start
        self.candidate_idx = candidate_idx
        self.label_idx = label_idx
        self.start_positions = start_positions
        self.end_positions = end_positions


class NodeInfo(object):
    def __init__(self,
                 start_position,
                 end_position,
                 node_idx):
        self.start_position = start_position
        self.end_position = end_position
        self.node_idx = node_idx


def build_graph(args, doc_tree, seq_len, la_start_positions, la_end_positions,
                idx_to_la_bound):
    graph = GraphIndoc(args)
    doc_node_idx = args.max_token + args.max_sentence + args.max_paragraph
    graph.add_node(doc_node_idx, -1)
    doc_node = NodeInfo(node_idx=doc_node_idx, start_position=0, end_position=0)
    para_nodes = []
    sent_nodes = []
    token_nodes = []
    para_attention_mask = []
    sent_attention_mask = []
    assert len(doc_tree) == args.max_indoc_paragraph * len(idx_to_la_bound), \
        '{}: {} * {}: {}: {}'.format(
        len(doc_tree), args.max_indoc_paragraph, len(idx_to_la_bound), idx_to_la_bound,
        doc_tree
    )
    for para_idx, (candidate_idx, paragraph) in enumerate(doc_tree):
        if len(para_nodes) < args.max_paragraph:
            para_node_idx = args.max_token + args.max_sentence + len(para_nodes)
            graph.add_node(para_node_idx, candidate_idx)
            para_nodes.append(
                NodeInfo(node_idx=para_node_idx, start_position=len(token_nodes),
                         end_position=len(token_nodes)))
        else:
            para_node_idx = -1
        e_pos = get_edge_position_indoc(EdgeType.PARAGRAPH_TO_DOCUMENT, len(para_nodes) - 1)
        graph.add_edge(para_node_idx, doc_node_idx, edge_type=EdgeType.PARAGRAPH_TO_DOCUMENT,
                       edge_pos=e_pos)
        for sent_idx, (sentence_idx, sentence) in enumerate(paragraph):
            if len(sent_nodes) < args.max_sentence:
                sent_node_idx = args.max_token + len(sent_nodes)
                graph.add_node(sent_node_idx, sentence_idx)
                sent_nodes.append(
                    NodeInfo(node_idx=sent_node_idx, start_position=len(token_nodes), end_position=len(token_nodes)))
            else:
                sent_node_idx = -1
            e_pos = get_edge_position_indoc(EdgeType.SENTENCE_TO_PARAGRAPH, len(sent_nodes) - 1)
            graph.add_edge(sent_node_idx, para_node_idx, edge_type=EdgeType.SENTENCE_TO_PARAGRAPH,
                           edge_pos=e_pos)

            for token_idx, (orig_tok_idx, global_token_pos) in enumerate(sentence):
                token_node_idx = global_token_pos
                graph.add_node(token_node_idx, orig_tok_idx)
                token_nodes.append(
                    NodeInfo(node_idx=token_node_idx, start_position=len(token_nodes), end_position=len(token_nodes)))

                e_pos = get_edge_position_indoc(EdgeType.TOKEN_TO_SENTENCE, token_idx)
                graph.add_edge(token_node_idx, sent_node_idx, edge_type=EdgeType.TOKEN_TO_SENTENCE,
                               edge_pos=e_pos)

            if sent_node_idx >= 0:
                sent_nodes[-1].end_position = len(token_nodes) - 1
        para_nodes[-1].start_position = idx_to_la_bound[candidate_idx]['start']
        para_nodes[-1].end_position = idx_to_la_bound[candidate_idx]['end']

    doc_node.end_position = len(token_nodes) - 1

    assert len(token_nodes) == seq_len, '***token_nodes: {}: {} ***'.format(
        len(token_nodes), seq_len
    )
    start_positions = []
    end_positions = []

    for la_start_position, la_end_position in zip(la_start_positions, la_end_positions):
        for para_idx, para_node in enumerate(para_nodes):
            if para_node.start_position <= la_start_position <= para_node.end_position:
                start_positions.append(para_idx)
            if para_node.start_position <= la_end_position <= para_node.end_position:
                end_positions.append(para_idx)
    assert len(la_start_positions) == len(start_positions)
    assert len(la_end_positions) == len(end_positions)
    for start_position, end_position in zip(start_positions, end_positions):
        assert start_position == end_position, '{}: {}'.format(
            start_position, end_position
        )

    sent_attention_mask = [True] * len(sent_nodes) + [False] * (args.max_sentence - len(sent_nodes))
    para_attention_mask = [True] * len(para_nodes) + [False] * (args.max_paragraph - len(para_nodes))
    graph_attention_mask = sent_attention_mask + para_attention_mask + [True]
    return graph, graph_attention_mask, start_positions, end_positions


def get_doc_tree(is_sentence_end, is_paragraph_end, orig_tok_idx, candidate_idx, input_mask):
    doc_len = len(is_sentence_end)
    assert len(is_paragraph_end) == doc_len
    assert len(orig_tok_idx) == doc_len
    assert len(candidate_idx) == doc_len
    assert len(input_mask) == doc_len
    document = []
    paragraph = []
    sentence = []
    cur_candidate_idx = -1
    for i in range(doc_len):
        if input_mask[i]:
            sentence.append((orig_tok_idx[i], i))
            cur_candidate_idx = max(cur_candidate_idx, candidate_idx[i])

            if is_sentence_end[i]:
                if candidate_idx[i] != -1:
                    paragraph.append((-1, sentence))
                sentence = []
            if is_paragraph_end[i]:
                assert len(sentence) == 0
                if candidate_idx[i] != -1:
                    document.append((cur_candidate_idx, paragraph))
                paragraph = []
                cur_candidate_idx = -1
    assert len(sentence) == 0
    assert len(paragraph) == 0
    return document


def chunk_to_feature(args, chunked_input_ids, chunked_input_mask, chunked_segment_ids,
                     chunked_candidate_idx, chunked_is_sentence_end,
                     chunked_is_paragraph_end, chunked_orig_tok_idx, is_training,
                     unique_id, example_id, last_doc_span_index, chunked_doc_span,
                     la_cand_tok_start_positions, la_cand_tok_end_positions,
                     la_tok_start_positions, la_tok_end_positions, query_token_lens,
                     all_doc_tokens):
    input_ids = []
    input_mask = []
    segment_ids = []
    candidate_idx = []
    is_sentence_end = []
    is_paragraph_end = []
    orig_tok_idx = []
    for ii, im, si, ci, ise, ipe, oti in zip(chunked_input_ids, chunked_input_mask,
                                             chunked_segment_ids,
                                             chunked_candidate_idx, chunked_is_sentence_end,
                                             chunked_is_paragraph_end, chunked_orig_tok_idx):
        input_ids += ii
        input_mask += im
        segment_ids += si
        candidate_idx += ci
        is_sentence_end += ise
        is_paragraph_end += ipe
        orig_tok_idx += oti

    seq_len = args.max_indoc_token * len(chunked_input_ids)
    assert len(input_ids) == seq_len
    assert len(input_mask) == seq_len
    assert len(segment_ids) == seq_len
    assert len(candidate_idx) == seq_len
    assert len(is_sentence_end) == seq_len
    assert len(is_paragraph_end) == seq_len
    assert len(orig_tok_idx) == seq_len
    doc_tree = get_doc_tree(is_sentence_end, is_paragraph_end, orig_tok_idx, candidate_idx,
                            input_mask)
    la_start_positions = []
    la_end_positions = []
    inchunk_start = 0
    idx_to_la_bound = {}
    for doc_span, query_token_len in zip(chunked_doc_span, query_token_lens):
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1

        for cand_idx, (la_cand_tok_start_position, la_cand_tok_end_position) in enumerate(zip(
            la_cand_tok_start_positions, la_cand_tok_end_positions
        )):
            doc_offset = query_token_len
            la_start_position = la_cand_tok_start_position - doc_start + doc_offset + inchunk_start
            la_end_position = la_cand_tok_end_position - doc_start + doc_offset + inchunk_start
            if la_cand_tok_start_position >= doc_start and la_cand_tok_end_position <= doc_end:
                idx_to_la_bound[cand_idx] = {'start': la_start_position,
                                             'end': la_end_position}
                break
        if is_training:
            for la_tok_start_position, la_tok_end_position in zip(la_tok_start_positions,
                                                                  la_tok_end_positions):
                if la_tok_start_position >= doc_start and la_tok_end_position <= doc_end:
                    doc_offset = query_token_len
                    la_start_position = la_tok_start_position - doc_start + doc_offset + inchunk_start
                    la_end_position = la_tok_end_position - doc_start + doc_offset + inchunk_start

                    assert la_start_position <= la_end_position
                    la_start_positions.append(la_start_position)
                    la_end_positions.append(la_end_position)
                elif la_tok_start_position >= doc_start and la_tok_start_position <= doc_end:
                    raise Exception('Cannot be in this condition!')
                elif doc_start > la_tok_start_position and doc_end < la_tok_end_position:
                    raise Exception('Cannot be in this condition!')
                elif doc_start <= la_tok_end_position and doc_end >= la_tok_end_position:
                    raise Exception('Cannot be in this condition!')

        inchunk_start += len(chunked_input_ids[0])

    seq_len = sum(segment_ids)
    graph, graph_attention_mask, start_positions, end_positions = \
        build_graph(args, doc_tree, seq_len,
                    la_start_positions, la_end_positions, idx_to_la_bound)
    if args.flag:
        args.flag = False
    if is_training:
        if len(start_positions) == 0:
            start_positions.append(-2)
            end_positions.append(-2)
        while len(start_positions) < NodePosition.MAX_EVIDENCE:
            assert len(start_positions) == len(end_positions)
            start_positions.append(start_positions[-1])
            end_positions.append(end_positions[-1])

    feature = InputFeatures(
        unique_id=unique_id,
        example_id=example_id,
        doc_span_index=last_doc_span_index,
        tokens=None,
        token_to_orig_map=None,
        token_is_max_context=None,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        graph=graph,
        graph_attention_mask=graph_attention_mask,
        start_positions=start_positions,
        is_paragraph_start=None,
        end_positions=end_positions,
        candidate_idx=candidate_idx,
        label_idx=None)
    return feature


def convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    """Loads a data file into a list of `InputBatch`s."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    unique_id = 1000000000

    docs = []
    for (example_index, example) in enumerate(examples):
        query_tokens = []
        for token in example.question_tokens:
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                query_tokens.append(sub_token)
        if len(query_tokens) > args.max_query_length:
            query_tokens = query_tokens[-args.max_query_length:]
        max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3
        sorted_la_candidates = sorted(enumerate(example.la_candidates), key=lambda x: x[1]["start_token"])
        tok_is_sentence_end = []
        tok_is_paragraph_end = []
        tok_candidate_idx = []
        tok_to_orig_index = []
        orig_to_tok_index = [-1] * len(example.doc_tokens)
        orig_to_tok_end_index = [-1] * len(example.doc_tokens)
        all_doc_tokens = []
        para_lens = []
        sents_lens = []
        for candidate_idx, candidate in sorted_la_candidates:
            start_index = candidate["start_token"]
            end_index = candidate["end_token"]
            context_tokens = []
            context_orig_idx = []
            new_end_index = None
            para_subtoken_len = 0
            for tok_idx in range(start_index, end_index):
                token_item = example.doc_tokens[tok_idx]
                para_subtoken_len += len(tokenizer.tokenize(token_item))
                if para_subtoken_len > max_tokens_for_doc:
                    new_end_index = tok_idx
                    if is_training:
                        for annotation in example.annotation:
                            la_start_token = annotation["long_answer"]["start_token"]
                            if la_start_token == start_index:
                                annotation["long_answer"]["end_token"] = new_end_index
                    candidate["end_token"] = new_end_index
                    break
                context_tokens.append(token_item.replace(" ", ""))
                if context_tokens[-1] == "":
                    context_tokens = context_tokens[:-1]
                    continue
                context_orig_idx.append(tok_idx)
            assert len(context_tokens) > 0

            paragraph_len = 0
            context = " ".join(context_tokens).strip()
            assert len(context) > 0
            context_sentences = nlp(context).sents
            context_idx = 0
            orig_to_tok_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
            sent_lens = []
            for sentence in context_sentences:
                sentence_len = 0
                for token in sentence.text.strip().split():
                    if len(context_tokens[context_idx]) == 0:
                        orig_to_tok_end_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
                        context_idx += 1
                        orig_to_tok_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
                    assert context_tokens[context_idx][:len(token)] == token
                    context_tokens[context_idx] = context_tokens[context_idx][len(token):]

                    sub_tokens = tokenizer.tokenize(token)
                    all_doc_tokens += sub_tokens
                    tok_to_orig_index += [context_orig_idx[context_idx]] * len(sub_tokens)
                    sentence_len += len(sub_tokens)
                    paragraph_len += len(sub_tokens)
                if sentence_len > 0:
                    tok_is_sentence_end += [False] * (sentence_len - 1) + [True]
                sent_lens.append(sentence_len)

            orig_to_tok_end_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
            assert context_idx + 1 == len(context_tokens)
            assert context_tokens[context_idx] == ""
            tok_is_paragraph_end += [False] * (paragraph_len - 1) + [True]
            tok_candidate_idx += [candidate_idx] * paragraph_len
            para_lens.append(paragraph_len)
            sents_lens.append(sent_lens)

        assert len(tok_is_sentence_end) == len(tok_is_paragraph_end), '{}: {}: {}: {}: {}'.format(
            len(tok_is_sentence_end), len(tok_is_paragraph_end), para_lens, sents_lens, example.example_id
        )
        assert len(tok_to_orig_index) == len(tok_is_sentence_end)
        assert len(para_lens) == len(sorted_la_candidates)

        la_tok_start_positions = []
        la_tok_end_positions = []
        la_cand_tok_start_positions = []
        la_cand_tok_end_positions = []
        for candidate_idx, candidate in sorted_la_candidates:
            start_index = candidate["start_token"]
            end_index = candidate["end_token"] - 1
            assert orig_to_tok_index[start_index] != -1
            assert orig_to_tok_end_index[end_index] != -1

            assert start_index <= end_index
            la_cand_tok_start_positions.append(orig_to_tok_index[start_index])
            la_cand_tok_end_positions.append(orig_to_tok_end_index[end_index] - 1)

        if is_training:
            for annot_idx, annotation in enumerate(example.annotation):
                la_start_token = annotation["long_answer"]["start_token"]
                la_end_token = annotation["long_answer"]["end_token"] - 1
                assert orig_to_tok_index[la_start_token] != -1
                assert orig_to_tok_end_index[la_end_token] != -1

                assert la_start_token <= la_end_token

                la_tok_start_positions.append(orig_to_tok_index[la_start_token])
                la_tok_end_positions.append(orig_to_tok_end_index[la_end_token] - 1)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        # max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        for p_idx, para_len in enumerate(para_lens):
            min_length = min(max_tokens_for_doc, para_len)
            doc_spans.append(_DocSpan(start=start_offset, length=min_length))
            start_offset += para_len
        assert start_offset == len(all_doc_tokens)

        doc = []
        chunked_input_ids = []
        chunked_input_mask = []
        chunked_segment_ids = []
        chunked_candidate_idx = []
        chunked_is_sentence_end = []
        chunked_is_paragraph_end = []
        chunked_orig_tok_idx = []
        chunked_doc_span = []
        query_token_lens = []
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            is_sentence_end = [True] + [False] * (len(tokens) - 3) + [True] + [True]
            is_paragraph_end = [True] + [False] * (len(tokens) - 3) + [True] + [True]
            orig_tok_idx = [-1] * len(tokens)
            candidate_idx = [-1] * len(tokens)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

                is_sentence_end.append(tok_is_sentence_end[split_token_index])
                is_paragraph_end.append(tok_is_paragraph_end[split_token_index])
                orig_tok_idx.append(tok_to_orig_index[split_token_index])
                candidate_idx.append(tok_candidate_idx[split_token_index])

            is_sentence_end[-1] = False
            is_paragraph_end[-1] = False
            tokens.append("[SEP]")
            segment_ids.append(1)
            is_sentence_end.append(True)
            is_paragraph_end.append(True)
            orig_tok_idx.append(-1)
            candidate_idx.append(candidate_idx[-1])

            assert len(orig_tok_idx) == len(tokens)
            assert len(is_sentence_end) == len(tokens)
            assert len(is_paragraph_end) == len(tokens)
            assert len(candidate_idx) == len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < args.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                is_sentence_end.append(False)
                is_paragraph_end.append(False)
                candidate_idx.append(-1)
                orig_tok_idx.append(-1)

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length
            assert len(is_sentence_end) == args.max_seq_length
            assert len(is_paragraph_end) == args.max_seq_length
            assert len(candidate_idx) == args.max_seq_length
            assert len(orig_tok_idx) == args.max_seq_length
            args.flag = False

            chunked_input_ids.append(input_ids)
            chunked_input_mask.append(input_mask)
            chunked_segment_ids.append(segment_ids)
            chunked_candidate_idx.append(candidate_idx)
            chunked_is_sentence_end.append(is_sentence_end)
            chunked_is_paragraph_end.append(is_paragraph_end)
            chunked_orig_tok_idx.append(orig_tok_idx)
            chunked_doc_span.append(doc_span)
            query_token_lens.append(len(query_tokens) + 2)

            if len(chunked_input_ids) == args.indoc_num:
                feature = chunk_to_feature(args=args, chunked_input_ids=chunked_input_ids,
                                           chunked_input_mask=chunked_input_mask,
                                           chunked_segment_ids=chunked_segment_ids,
                                           chunked_is_paragraph_end=chunked_is_paragraph_end,
                                           chunked_candidate_idx=chunked_candidate_idx,
                                           chunked_is_sentence_end=chunked_is_sentence_end,
                                           chunked_orig_tok_idx=chunked_orig_tok_idx,
                                           chunked_doc_span=chunked_doc_span,
                                           is_training=is_training, unique_id=unique_id,
                                           example_id=example.example_id, last_doc_span_index=doc_span_index,
                                           la_tok_start_positions=la_tok_start_positions,
                                           la_tok_end_positions=la_tok_end_positions,
                                           la_cand_tok_start_positions=la_cand_tok_start_positions,
                                           la_cand_tok_end_positions=la_cand_tok_end_positions,
                                           query_token_lens=query_token_lens,
                                           all_doc_tokens=all_doc_tokens)
                doc.append(feature)
                chunked_input_ids = []
                chunked_input_mask = []
                chunked_segment_ids = []
                chunked_is_paragraph_end = []
                chunked_candidate_idx = []
                chunked_is_sentence_end = []
                chunked_orig_tok_idx = []
                chunked_doc_span = []
                query_token_lens = []
            unique_id += 1

        if len(chunked_input_ids) > 0:
            feature = chunk_to_feature(args=args, chunked_input_ids=chunked_input_ids,
                                       chunked_input_mask=chunked_input_mask,
                                       chunked_segment_ids=chunked_segment_ids,
                                       chunked_is_paragraph_end=chunked_is_paragraph_end,
                                       chunked_candidate_idx=chunked_candidate_idx,
                                       chunked_is_sentence_end=chunked_is_sentence_end,
                                       chunked_orig_tok_idx=chunked_orig_tok_idx,
                                       chunked_doc_span=chunked_doc_span,
                                       is_training=is_training, unique_id=unique_id,
                                       example_id=example.example_id,
                                       last_doc_span_index=doc_span_index,
                                       la_tok_start_positions=la_tok_start_positions,
                                       la_tok_end_positions=la_tok_end_positions,
                                       la_cand_tok_start_positions=la_cand_tok_start_positions,
                                       la_cand_tok_end_positions=la_cand_tok_end_positions,
                                       query_token_lens=query_token_lens,
                                       all_doc_tokens=all_doc_tokens)
            doc.append(feature)
            chunked_input_ids = []
            chunked_input_mask = []
            chunked_segment_ids = []
            chunked_is_paragraph_end = []
            chunked_candidate_idx = []
            chunked_is_sentence_end = []
            chunked_orig_tok_idx = []
            chunked_doc_span = []
            query_token_lens = []
        docs.append(doc)

    logging.info("  Saving features into cached file {}".format(cached_path))
    with open(cached_path, "wb") as writer:
        pickle.dump(docs, writer)

    return cached_path


def run_convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    pool = []
    p = multiprocessing.Pool(args.num_threads)
    for i in range(args.num_threads):
        start_index = len(examples) // args.num_threads * i
        end_index = len(examples) // args.num_threads * (i + 1)
        if i == args.num_threads - 1:
            end_index = len(examples)
        pool.append(p.apply_async(convert_examples_to_features, args=(
            args, examples[start_index: end_index], tokenizer, is_training, cached_path + ".part" + str(i))))
    p.close()
    p.join()

    features = []
    for i, thread in enumerate(pool):
        cached_path_tmp = thread.get()
        logging.info("Reading thread {} output from {}".format(i, cached_path_tmp))
        with open(cached_path_tmp, "rb") as reader:
            features_tmp = pickle.load(reader)
        if not args.keep_parts:
            os.remove(cached_path_tmp)
        features += features_tmp

    max_features_cnt = 0
    mean_features_cnt = []
    for doc in features:
        max_features_cnt = max(max_features_cnt, len(doc))
        mean_features_cnt.append(len(doc))
    logging.info("  Max feature number in a doc is {}".format(max_features_cnt))
    logging.info("  Mean feature number in a doc is {}".format(float(sum(mean_features_cnt)) / len(mean_features_cnt)))

    logging.info("  Saving features from into cached file {0}".format(cached_path))
    with open(cached_path, "wb") as writer:
        pickle.dump(features, writer)


def text_clean(text):
    if isinstance(text, str):
        cleaned_txt = text.replace("\n", " ").strip()
        cleaned_txt = ' '.join(nltk.word_tokenize(cleaned_txt))
        cleaned_txt = cleaned_txt.strip()
        return cleaned_txt if cleaned_txt else '[ PAD ] .'
    elif isinstance(text, list):
        cleaned_txts = []
        for txt in text:
            if isinstance(txt, str):
                cleaned_txt = txt.replace("\n", " ").strip()
                cleaned_txt = ' '.join(nltk.word_tokenize(cleaned_txt))
                cleaned_txt = cleaned_txt.strip()
                cleaned_txts.append(cleaned_txt if cleaned_txt else '[ PAD ] .')
            else:
                print('Warning: find nonstr, convert to empty str')
                cleaned_txts.append('[ PAD ] .')
        return cleaned_txts
    else:
        print('Warning: find nonstr, convert to empty str')
        return '[ PAD ] .'


def convert_qasper(input_data):
    standard_dicts = []
    for _, article in tqdm(input_data.items(), total=len(input_data), desc='read data...'):
        id_to_p = []
        is_secname = []
        for section_info in article['full_text']:
            section_name = text_clean(section_info['section_name'])
            assert len(section_name) > 0
            id_to_p.append(section_name)
            is_secname.append(True)
            for p in text_clean(section_info['paragraphs']):
                assert len(p) > 0
                id_to_p.append(p)
                is_secname.append(False)
        for qa in article['qas']:
            sample = {}
            sample['qid'] = qa['question_id']
            sample['query'] = qa['question']
            sample['paragraphs'] = copy.deepcopy(id_to_p)
            sample['is_secname'] = copy.deepcopy(is_secname)
            if not sample['paragraphs']:
                sample['paragraphs'] = ["[ PAD ] ."]
                sample['is_secname'] = [True]
            gold = [0] * len(sample['paragraphs'])
            for ans in qa['answers']:
                answer = ans['answer']
                for evi in answer['evidence']:
                    if 'FLOAT SELECTED:' in evi or evi == '':
                        continue
                    evi = text_clean(evi)
                    try:
                        gold[sample['paragraphs'].index(evi)] = 1
                    except Exception as e:
                        print(e)
                        print('***sample[\'paragraphs\']: {}***'.format(sample['paragraphs']))
                        raise Exception(e)
            sample['gold'] = gold
            assert len(sample['gold']) == len(sample['paragraphs']), '{}: {}'.format(
                len(sample['gold']), len(sample['paragraphs'])
            )
            assert len(sample['gold']) == len(sample['is_secname']), '{}: {}'.format(
                len(sample['gold']), len(sample['is_secname'])
            )
            for p in sample['paragraphs']:
                assert isinstance(p, str), '{}'.format(sample)
            for label in sample['gold']:
                assert label == 0 or label == 1
            standard_dicts.append(sample)

    examples = []
    for sample in standard_dicts:
        example_id = sample['qid']
        question_tokens = sample['query'].split()
        doc_tokens = []
        la_candidates = []
        annotations = []
        start_token = 0
        for p_idx, (para, label) in enumerate(zip(sample['paragraphs'], sample['gold'])):
            para_tokens = para.split()
            doc_tokens.extend(para_tokens)
            end_token = start_token + len(para_tokens)
            assert len(doc_tokens) == end_token
            la_candidates.append({
                'start_token': start_token,
                'end_token': end_token,
                'candidate_index': p_idx
            })
            if label:
                annotations.append({
                    'long_answer': {
                        'start_token': start_token,
                        'end_token': end_token,
                        'candidate_index': p_idx
                    }
                })
            start_token = end_token
        examples.append({
            'question_tokens': question_tokens,
            'document_tokens': doc_tokens,
            'annotations': annotations,
            'example_id': example_id,
            'long_answer_candidates': la_candidates
        })

    return examples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--max_para_len", default=256, type=int)
    parser.add_argument("--num_threads", default=16, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--is_training", action='store_true')
    parser.add_argument("--keep_parts", action='store_true',
                        help='whether or not remove the intermedia file for each process')
    parser.add_argument("--my_config", default=None, type=str, required=True)
    parser.add_argument("--indoc_num", type=int, default=None)

    args = parser.parse_args()
    my_config = Config(args.my_config)
    args.max_indoc_token = my_config.max_indoc_token_len
    args.max_indoc_sentence = my_config.max_indoc_sentence_len
    args.max_indoc_paragraph = my_config.max_indoc_paragraph_len
    args.indoc_num = my_config.indoc_num if args.indoc_num is None else args.indoc_num
    args.max_token = args.max_indoc_token * args.indoc_num
    args.max_sentence = args.max_indoc_sentence * args.indoc_num
    args.max_paragraph = args.max_indoc_paragraph * args.indoc_num
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    prefix = "cached_{0}_{1}_{2}_{3}_indoc{4}".format(list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                                      str(args.max_seq_length),
                                                      str(args.doc_stride),
                                                      str(args.max_query_length),
                                                      str(args.indoc_num))
    prefix += '_train' if args.is_training else '_test'

    prefix = os.path.join(args.output_dir, prefix)
    os.makedirs(prefix, exist_ok=True)

    input_path = args.input_path
    cached_path = os.path.join(prefix, os.path.split(input_path)[1] + ".pkl")
    is_training = args.is_training

    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        logging.info("Reading data from {}.".format(input_path))
        input_data = json.load(f)
        qpapers = convert_qasper(input_data)
        for json_example in qpapers:

            question_tokens = json_example["question_tokens"]
            doc_tokens = json_example["document_tokens"]
            annotation = json_example["annotations"]
            example_id = json_example["example_id"]
            la_candidates = json_example["long_answer_candidates"]

            examples.append(
                Example(example_id=example_id,
                        question_tokens=question_tokens,
                        doc_tokens=doc_tokens,
                        la_candidates=la_candidates,
                        annotation=annotation))

    run_convert_examples_to_features(args=args,
                                     examples=examples,
                                     tokenizer=tokenizer,
                                     is_training=is_training,
                                     cached_path=cached_path)


if __name__ == "__main__":
    main()
