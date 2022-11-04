import enum
import math
import copy
import json
import torch
import collections
import pickle
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EdgeType(enum.IntEnum):
    TOKEN_TO_TOKEN = 0
    TOKEN_TO_SENTENCE = 1
    TOKEN_TO_PARAGRAPH = 2
    TOKEN_TO_DOCUMENT = 3

    SENTENCE_TO_TOKEN = 4
    SENTENCE_TO_SENTENCE = 5
    SENTENCE_TO_PARAGRAPH = 6
    SENTENCE_TO_DOCUMENT = 7

    PARAGRAPH_TO_TOKEN = 8
    PARAGRAPH_TO_SENTENCE = 9
    PARAGRAPH_TO_PARAGRAPH = 10
    PARAGRAPH_TO_DOCUMENT = 11

    DOCUMENT_TO_TOKEN = 12
    DOCUMENT_TO_SENTENCE = 13
    DOCUMENT_TO_PARAGRAPH = 14
    DOCUMENT_TO_DOCUMENT = 15


class NodeType(enum.IntEnum):
    TOKEN = 0
    SENTENCE = 1
    PARAGRAPH = 2
    DOCUMENT = 3


class NodePosition(enum.IntEnum):
    MAX_PARAGRAPH = 64
    MAX_SENTENCE = 128
    MAX_TOKEN = 512
    MAX_EVIDENCE = 24


class MetaNodePosition():
    MAX_PARAGRAPH = 32
    MAX_SENTENCE = 64
    MAX_CHUNK = 4
    def __init__(self, indoc_num=16, max_sentence=None, max_paragraph=None, max_chunk=None):
        self.indoc_num = indoc_num
        MetaNodePosition.MAX_SENTENCE = indoc_num * 4
        MetaNodePosition.MAX_PARAGRAPH = indoc_num * 2
        MetaNodePosition.MAX_CHUNK = indoc_num // 4 if indoc_num >= 4 else 1
        if max_sentence is not None:
            MetaNodePosition.MAX_SENTENCE = max_sentence
        if max_paragraph is not None:
            MetaNodePosition.MAX_PARAGRAPH = max_paragraph
        if max_chunk is not None:
            MetaNodePosition.MAX_CHUNK = max_chunk


class EdgePositionIndoc():
    
    NUM_TOKEN_TO_TOKEN = 0
    NUM_TOKEN_TO_SENTENCE = 64 * 16
    NUM_TOKEN_TO_PARAGRAPH = 256 * 16
    NUM_TOKEN_TO_DOCUMENT = 512 * 16

    NUM_SENTENCE_TO_TOKEN = 64 * 16
    NUM_SENTENCE_TO_SENTENCE = 0 * 16
    NUM_SENTENCE_TO_PARAGRAPH = 32 * 16
    NUM_SENTENCE_TO_DOCUMENT = 64 * 16

    NUM_PARAGRAPH_TO_TOKEN = 256 * 16
    NUM_PARAGRAPH_TO_SENTENCE = 32 * 16
    NUM_PARAGRAPH_TO_PARAGRAPH = 0 * 16
    NUM_PARAGRAPH_TO_DOCUMENT = 32 * 16

    NUM_DOCUMENT_TO_TOKEN = 512 * 16
    NUM_DOCUMENT_TO_SENTENCE = 64 * 16
    NUM_DOCUMENT_TO_PARAGRAPH = 32 * 16
    NUM_DOCUMENT_TO_DOCUMENT = 0 * 16

    max_edge_types = [NUM_TOKEN_TO_TOKEN, NUM_TOKEN_TO_SENTENCE, NUM_TOKEN_TO_PARAGRAPH, NUM_TOKEN_TO_DOCUMENT,
                      NUM_SENTENCE_TO_TOKEN, NUM_SENTENCE_TO_SENTENCE, NUM_SENTENCE_TO_PARAGRAPH,
                      NUM_SENTENCE_TO_DOCUMENT, NUM_PARAGRAPH_TO_TOKEN, NUM_PARAGRAPH_TO_SENTENCE,
                      NUM_PARAGRAPH_TO_PARAGRAPH, NUM_PARAGRAPH_TO_DOCUMENT, NUM_DOCUMENT_TO_TOKEN,
                      NUM_DOCUMENT_TO_SENTENCE, NUM_DOCUMENT_TO_PARAGRAPH, NUM_DOCUMENT_TO_DOCUMENT]

    def __init__(self, indoc_num=16):
        self.indoc_num = indoc_num
        EdgePositionIndoc.NUM_TOKEN_TO_TOKEN = 0
        EdgePositionIndoc.NUM_TOKEN_TO_SENTENCE = 64 * indoc_num
        EdgePositionIndoc.NUM_TOKEN_TO_PARAGRAPH = 256 * indoc_num
        EdgePositionIndoc.NUM_TOKEN_TO_DOCUMENT = 512 * indoc_num

        EdgePositionIndoc.NUM_SENTENCE_TO_TOKEN = 64 * indoc_num
        EdgePositionIndoc.NUM_SENTENCE_TO_SENTENCE = 0 * indoc_num
        EdgePositionIndoc.NUM_SENTENCE_TO_PARAGRAPH = 32 * indoc_num
        EdgePositionIndoc.NUM_SENTENCE_TO_DOCUMENT = 64 * indoc_num

        EdgePositionIndoc.NUM_PARAGRAPH_TO_TOKEN = 256 * indoc_num
        EdgePositionIndoc.NUM_PARAGRAPH_TO_SENTENCE = 32 * indoc_num
        EdgePositionIndoc.NUM_PARAGRAPH_TO_PARAGRAPH = 0 * indoc_num
        EdgePositionIndoc.NUM_PARAGRAPH_TO_DOCUMENT = 32 * indoc_num

        EdgePositionIndoc.NUM_DOCUMENT_TO_TOKEN = 512 * indoc_num
        EdgePositionIndoc.NUM_DOCUMENT_TO_SENTENCE = 64 * indoc_num
        EdgePositionIndoc.NUM_DOCUMENT_TO_PARAGRAPH = 32 * indoc_num
        EdgePositionIndoc.NUM_DOCUMENT_TO_DOCUMENT = 0 * indoc_num
        EdgePositionIndoc.edge_type_start = [0]
        for edge_type in range(1, 16):
            EdgePositionIndoc.edge_type_start.append(
                EdgePositionIndoc.edge_type_start[edge_type - 1] + EdgePositionIndoc.max_edge_types[edge_type - 1])


EdgePositionIndoc.edge_type_start = [0]
for edge_type in range(1, 16):
    EdgePositionIndoc.edge_type_start.append(
        EdgePositionIndoc.edge_type_start[edge_type - 1] + EdgePositionIndoc.max_edge_types[edge_type - 1])


def get_edge_position_indoc(edge_type, edge_idx):
    return EdgePositionIndoc.edge_type_start[edge_type] + min(edge_idx, EdgePositionIndoc.max_edge_types[edge_type] - 1)


class GraphIndoc(object):
    def __init__(self, args):
        self.indoc_num = args.indoc_num
        self.max_token = self.indoc_num * args.max_indoc_token
        self.max_sentence = self.indoc_num * args.max_indoc_sentence
        self.max_paragraph = self.indoc_num * args.max_indoc_paragraph
        self.edges_src = []
        self.edges_tgt = []
        self.edges_type = []
        self.edges_pos = []
        self.st_mask = [0] * (self.max_token + self.max_sentence + self.max_paragraph + 1)
        self.st_index = [-1] * (self.max_token + self.max_sentence + self.max_paragraph + 1)

    def add_node(self, idx, index=-1):
        self.st_mask[idx] = 1
        self.st_index[idx] = index

    def add_edge(self, src, tgt, edge_type=-1, edge_pos=-1):
        if src < 0 or tgt < 0:
            return

        assert self.st_mask[src] > 0 and self.st_mask[tgt] > 0
        self.edges_src.append(src)
        self.edges_tgt.append(tgt)
        self.edges_type.append(edge_type)
        self.edges_pos.append(edge_pos)
        assert edge_pos >= 0, 'src: {}, tgt: {}, edge_type: {}, edge_pos: {}'.format(
            src, tgt, edge_type, edge_pos
        )


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Config(object):
    def __init__(self, config_json_file):
        with open(config_json_file, "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class AttentionOutputLayer(nn.Module):
    def __init__(self, config, in_dim=2):
        super(AttentionOutputLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size * in_dim, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class GraphRNNIndoc(nn.Module):
    def __init__(self, config):
        super(GraphRNNIndoc, self).__init__()
        self.birnn = nn.LSTM(config.hidden_size,
                             config.hidden_size // 2,
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True)

    def forward(self, hidden_states, st_mask):
        flat_text_embeddings = hidden_states.view(-1, hidden_states.size(-1))
        flat_text_mask = st_mask.view(-1).byte()
        filtered_text_embeddings = flat_text_embeddings[flat_text_mask.bool()]
        filtered_contextualized_embeddings, _ = self.birnn(filtered_text_embeddings.unsqueeze(0))
        filtered_contextualized_embeddings = filtered_contextualized_embeddings.squeeze(0)
        flat_contextualized_embeddings = torch.zeros((flat_text_embeddings.size(0),
                                                      filtered_contextualized_embeddings.size(-1)),
                                                     device=filtered_text_embeddings.device)
        flat_contextualized_embeddings = flat_contextualized_embeddings.masked_scatter(
            flat_text_mask.unsqueeze(-1).bool(), filtered_contextualized_embeddings
        )  # all_sequence
        contextualized_embeddings = flat_contextualized_embeddings.reshape(
            (hidden_states.size(0), hidden_states.size(1), flat_contextualized_embeddings.size(-1))
        )
        encoder_outputs = contextualized_embeddings
        return encoder_outputs


class IntegrationLayer(nn.Module):
    def __init__(self, config):
        super(IntegrationLayer, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.num_edge_types = config.num_edge_types

    def forward(self, hidden_states, edges):
        edges_src, edges_tgt, edges_type, edges_pos = edges
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        query_layer = self.query(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                     self.attention_head_size)
        key_layer = self.key(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                 self.attention_head_size)
        value_layer = self.value(hidden_states).view(batch_size * seq_len, self.num_attention_heads,
                                                     self.attention_head_size)
        src_key_tensor = key_layer[edges_src]

        tgt_query_tensor = query_layer[edges_tgt]

        # (n_edges, n_heads)
        attention_scores = torch.exp((tgt_query_tensor * src_key_tensor).sum(-1) / math.sqrt(self.attention_head_size))

        sum_attention_scores = hidden_states.data.new(batch_size * seq_len, self.num_attention_heads).fill_(0)
        indices = edges_tgt.view(-1, 1).expand(-1, self.num_attention_heads)
        sum_attention_scores.scatter_add_(dim=0, index=indices, src=attention_scores)

        attention_scores = attention_scores / sum_attention_scores[edges_tgt]
        # (n_edges, n_heads, head_size) * (n_edges, n_heads, 1)
        src_value_tensor = value_layer[edges_src]

        src_value_tensor *= attention_scores.unsqueeze(-1)

        output = hidden_states.data.new(
            batch_size * seq_len, self.num_attention_heads, self.attention_head_size).fill_(0)
        indices = edges_tgt.view(-1, 1, 1).expand(-1, self.num_attention_heads, self.attention_head_size)
        output.scatter_add_(dim=0, index=indices, src=src_value_tensor)
        output = output.view(batch_size, seq_len, -1)

        return hidden_states, output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_key = nn.Linear(input_dim, output_dim)
        self.linear_value = nn.Linear(input_dim, output_dim)
        self.linear_query = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, head_num, mask=None):
        '''
        q/k/v:
        mask: batch_size, q_len, k_len
        '''
        # [batch_size, steps, head_num, per_head_dim]
        batch_size = query.shape[0]

        q_len = query.shape[1]
        k_len = key.shape[1]
        v_len = value.shape[1]

        per_head_dim = self.output_dim // head_num

        query_ = self.linear_query(query)
        key_ = self.linear_key(key)
        value_ = self.linear_value(value)

        query_ = torch.reshape(query_, (batch_size, head_num, q_len, per_head_dim))
        key_ = torch.reshape(key_, (batch_size, head_num, per_head_dim, k_len))
        value_ = torch.reshape(value_, (batch_size, head_num, v_len, per_head_dim))

        score = torch.matmul(query_, key_)  # batch_size, head_num, q_len, k_len
        if mask is not None:  # [batch_size, seq_len]
            mask_score = torch.reshape(mask.float(), (batch_size, 1, 1, -1))
            mask_score = (1 - mask_score.float()) * -99999.0

            score = mask_score + score

        score = torch.nn.functional.softmax(score, -1)
        score = self.dropout(score)  # strange dropout in Transformer

        outs = torch.matmul(score, value_)  # batch_size, head_num, q_len, per_head_dim
        outs = torch.transpose(outs, 1, 2)
        outs = torch.reshape(outs, (batch_size, q_len, head_num * per_head_dim))

        return outs


class MetaGraphAttention(nn.Module):
    def __init__(self, config):
        super(MetaGraphAttention, self).__init__()
        self.memory_output = AttentionOutputLayer(config)
        self.hidden_size = config.hidden_size
        self.gat_head_num = 6
        self.gat_dropout = 0.1
        self.hops = config.meta_gat_hops if config.meta_gat_hops is not None else 1
        self.meta_attn_graph = MultiHeadAttention(
            input_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.gat_dropout
        )

    def forward(self, meta_hiddens):
        node_types = [NodeType.SENTENCE, NodeType.PARAGRAPH, NodeType.DOCUMENT]
        meta_node_starts = [0, MetaNodePosition.MAX_SENTENCE, MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH]
        meta_node_lens = [MetaNodePosition.MAX_SENTENCE, MetaNodePosition.MAX_PARAGRAPH, MetaNodePosition.MAX_CHUNK]
        memory_outputs = [meta_hiddens]
        for hop_idx in range(self.hops):
            metas_src2tgt_g = [None for _ in node_types]
            metas_tgt2src_g = [None for _ in node_types]
            for node_idx, node_type in enumerate(node_types):
                src_node_idx = node_idx
                tgt_node_idx = node_idx + 1 if node_idx + 1 < len(node_types) else 0
                src_node_start = meta_node_starts[src_node_idx]
                src_node_len = meta_node_lens[src_node_idx]
                tgt_node_start = meta_node_starts[tgt_node_idx]
                tgt_node_len = meta_node_lens[tgt_node_idx]
                meta_src2tgt_g = self.meta_attn_graph(meta_hiddens[:, tgt_node_start: tgt_node_start + tgt_node_len, :],
                                                      memory_outputs[-1][:, src_node_start: src_node_start + src_node_len, :],
                                                      memory_outputs[-1][:, src_node_start: src_node_start + src_node_len, :],
                                                      self.gat_head_num)
                meta_tgt2src_g = self.meta_attn_graph(meta_hiddens[:, src_node_start: src_node_start + src_node_len, :],
                                                      memory_outputs[-1][:, tgt_node_start: tgt_node_start + tgt_node_len, :],
                                                      memory_outputs[-1][:, tgt_node_start: tgt_node_start + tgt_node_len, :],
                                                      self.gat_head_num)
                metas_src2tgt_g[tgt_node_idx] = meta_src2tgt_g
                metas_tgt2src_g[src_node_idx] = meta_tgt2src_g
            metas_src2tgt_g = torch.cat(metas_src2tgt_g, dim=-2)
            metas_tgt2src_g = torch.cat(metas_tgt2src_g, dim=-2)
            memory_g = torch.cat([metas_src2tgt_g, metas_tgt2src_g], dim=-1)
            memory_output = self.memory_output(memory_g, memory_outputs[-1])
            memory_outputs.append(memory_output)
        return memory_outputs[-1]


class MetaNodeInteractionLayerV2Indoc(nn.Module):
    def __init__(self, config):
        super(MetaNodeInteractionLayerV2Indoc, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.memory_slots = MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH \
            + MetaNodePosition.MAX_CHUNK
        self.memory_dim = config.hidden_size
        self.memory_hops = config.memory_hops
        self.hidden_size = config.hidden_size
        self.meta_graph_attention = MetaGraphAttention(config)
        self.gat_head_num = 6
        self.gat_dropout = 0.1
        self.initial_memory = nn.Parameter(torch.normal(mean=0,
                                                        std=1,
                                                        size=(self.memory_slots,
                                                              self.memory_dim)))
        self.attn_graph = MultiHeadAttention(
            input_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.gat_dropout
        )
        self.hidden_output = AttentionOutputLayer(config)
        self.mem_merge = nn.Sequential(collections.OrderedDict([
            ('converter_dense_0', torch.nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ('converter_tanh_0', torch.nn.Tanh())
        ]))
        self.u_gate = nn.Sequential(collections.OrderedDict([
            ('u_dense_0', torch.nn.Linear(self.hidden_size * 2, 1)),
            ('u_sigmoid_0', torch.nn.Sigmoid())
        ]))

    def forward(self, hidden_states, meta_hiddens, graph_attention_mask, cur_config):
        max_token = cur_config['max_token']
        max_sentence = cur_config['max_sentence']
        sentence_start = cur_config['sentence_start']
        max_paragraph = cur_config['max_paragraph']
        batch_size = hidden_states.size(0)
        if meta_hiddens is None:
            meta_hiddens = self.initial_memory.unsqueeze(0).repeat(batch_size, 1, 1)
        graph_states = [(hidden_states, meta_hiddens), ]
        node_types = [NodeType.SENTENCE, NodeType.PARAGRAPH, NodeType.DOCUMENT]
        node_starts = [sentence_start, sentence_start + max_sentence,
                       sentence_start + max_sentence + max_paragraph]
        node_lens = [max_sentence, max_paragraph, 1]
        meta_node_starts = [0, MetaNodePosition.MAX_SENTENCE, MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH]
        meta_node_lens = [MetaNodePosition.MAX_SENTENCE, MetaNodePosition.MAX_PARAGRAPH, MetaNodePosition.MAX_CHUNK]
        for i in range(self.memory_hops):
            parts_hidden_states_g = []
            parts_hidden_states_g.append(torch.zeros_like(graph_states[i][0][:, 0: sentence_start, :]))
            parts_memory_g = []
            for node_type, node_start, node_len, meta_node_start, meta_node_len in \
                    zip(node_types, node_starts, node_lens, meta_node_starts, meta_node_lens):
                with torch.no_grad():
                    graph_node_start = node_start - sentence_start
                    part_memory_g = self.attn_graph(graph_states[i][1][:, meta_node_start: meta_node_start + meta_node_len, :],
                                                    graph_states[i][0][:, node_start: node_start + node_len, :],
                                                    graph_states[i][0][:, node_start: node_start + node_len, :],
                                                    self.gat_head_num,
                                                    mask=graph_attention_mask[:, graph_node_start: graph_node_start + node_len])
                parts_memory_g.append(part_memory_g)
            memory_g = torch.cat(parts_memory_g, dim=-2)
            mem_combined = torch.cat([meta_hiddens, memory_g], dim=-1)
            h_prime = self.mem_merge(mem_combined)
            z = self.u_gate(mem_combined)
            h_present = (1 - z) * meta_hiddens + z * h_prime
            memory_output = h_present
            memory_gat = self.meta_graph_attention(memory_output)
            for node_type, node_start, node_len, meta_node_start, meta_node_len in \
                    zip(node_types, node_starts, node_lens, meta_node_starts, meta_node_lens):
                part_hidden_states_g = self.attn_graph(graph_states[i][0][:, node_start: node_start + node_len, :],
                                                       memory_gat[:, meta_node_start: meta_node_start + meta_node_len, :],
                                                       memory_gat[:, meta_node_start: meta_node_start + meta_node_len, :],
                                                       self.gat_head_num)
                parts_hidden_states_g.append(part_hidden_states_g)
            hidden_states_g = torch.cat(parts_hidden_states_g, dim=-2)
            graph_states.append((hidden_states_g, memory_output))

        hidden_states_g = graph_states[-1][0]
        memory_output = graph_states[-1][1]

        merge_hidden_states = torch.cat([hidden_states, hidden_states_g], -1)
        hidden_output = self.hidden_output(merge_hidden_states, hidden_states)
        return hidden_output, memory_output


class MetaGraphAttentionLayerIndoc(nn.Module):
    def __init__(self, config):
        super(MetaGraphAttentionLayerIndoc, self).__init__()
        self.token_attention = GraphRNNIndoc(config)
        self.sentence_attention = GraphRNNIndoc(config)
        self.paragraph_attention = GraphRNNIndoc(config)
        self.output_layer = AttentionOutputLayer(config, in_dim=1)
        self.config = config
        self.integration = IntegrationLayer(config)
        self.meta_integration = MetaNodeInteractionLayerV2Indoc(config)
        self.output = AttentionOutputLayer(config, in_dim=1)

    def forward(self, input_tensor, meta_hiddens, graph_attention_mask, st_mask, edges, segment_ids,
                cur_config):
        # input_tensor: [1, (max_indoc_token + max_indoc_sent + max_indoc_para) * indoc_num + 1]
        max_token = cur_config['max_token']
        max_sentence = cur_config['max_sentence']
        sentence_start = cur_config['sentence_start']
        max_paragraph = cur_config['max_paragraph']
        batch_size = st_mask.size(0)
        assert batch_size == 1
        hidden_size = input_tensor.size(-1)
        chunk_tensor = input_tensor[:, -1:, :]
        _input_tensor = input_tensor[:, :-1, :]
        _input_tensor = _input_tensor.view(batch_size, -1, hidden_size)
        _segment_ids = segment_ids.view(1, -1)
        assert _segment_ids.size(1) == max_token
        token_mask = torch.logical_and(st_mask[:, :max_token].bool(), _segment_ids.bool())
        token_output = self.token_attention(_input_tensor[:, :max_token, :], token_mask)
        sent_start = sentence_start
        sent_end = sent_start + max_sentence
        sentence_output = self.sentence_attention(_input_tensor[:, sent_start:sent_end, :],
                                                  st_mask[:, sent_start:sent_end])

        para_start = sent_end
        para_end = sent_end + max_paragraph
        paragraph_output = self.paragraph_attention(_input_tensor[:, para_start:para_end, :],
                                                    st_mask[:, para_start:para_end])
        hiddens_output = torch.cat([token_output, _input_tensor[:, max_token: sentence_start, :],
                                    sentence_output, paragraph_output, chunk_tensor], dim=1)
        meta_hiddens_output = None
        for _ in range(self.config.neighborhops):
            hiddens, output = self.integration(hiddens_output, edges)
            hiddens_output = self.output_layer(hidden_states=output, input_tensor=hiddens)
        hiddens_output = self.output(hiddens_output, input_tensor)
        hiddens_output, meta_hiddens_output = self.meta_integration(hidden_states=hiddens_output,
                                                                    meta_hiddens=meta_hiddens,
                                                                    graph_attention_mask=graph_attention_mask,
                                                                    cur_config=cur_config)
        return hiddens_output, meta_hiddens_output


class IntermediateLayer(nn.Module):
    def __init__(self, config):
        super(IntermediateLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class MetaEncoderLayerIndoc(nn.Module):
    def __init__(self, config):
        super(MetaEncoderLayerIndoc, self).__init__()
        self.attention = MetaGraphAttentionLayerIndoc(config)
        self.intermediate = IntermediateLayer(config)
        self.output = OutputLayer(config)

    def forward(self, hidden_states, meta_hiddens, graph_attention_mask, st_mask, edges, segment_ids,
                cur_config):
        attention_output, meta_hiddens = self.attention(input_tensor=hidden_states,
                                                        meta_hiddens=meta_hiddens,
                                                        graph_attention_mask=graph_attention_mask,
                                                        st_mask=st_mask, edges=edges,
                                                        segment_ids=segment_ids,
                                                        cur_config=cur_config)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        layer_output = attention_output
        return layer_output, meta_hiddens


class MetaEncoderIndoc(nn.Module):
    def __init__(self, config):
        super(MetaEncoderIndoc, self).__init__()
        self.initializer = InitializerIndoc(config)
        layer = MetaEncoderLayerIndoc(config)
        self.config = config
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, meta_hiddens, graph_attention_mask, st_mask,
                edges, segment_ids, cur_config, output_all_encoded_layers=True):
        # return: hidden_states [1, (max_indoc_token + max_indoc_sent + max_indoc_para) * indoc_num + 1]
        hidden_states = self.initializer(hidden_states, st_mask, edges, cur_config)
        all_encoder_layers = []
        all_meta_hiddens = []
        for layer_module in self.layer:
            hidden_states, meta_hiddens = layer_module(hidden_states=hidden_states,
                                                       meta_hiddens=meta_hiddens,
                                                       graph_attention_mask=graph_attention_mask,
                                                       st_mask=st_mask, edges=edges,
                                                       segment_ids=segment_ids,
                                                       cur_config=cur_config)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_meta_hiddens.append(meta_hiddens)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_meta_hiddens.append(meta_hiddens)
        return all_encoder_layers, all_meta_hiddens


class InitializerIndoc(nn.Module):
    def __init__(self, config):
        super(InitializerIndoc, self).__init__()
        self.max_indoc_token = config.max_indoc_token_len
        max_sentence = config.max_indoc_sentence_len * config.indoc_num
        max_paragraph = config.max_indoc_paragraph_len * config.indoc_num
        self.position_embeddings = nn.Embedding(max_sentence + max_paragraph + 1,
                                                config.hidden_size)
        self.config = config

    def forward(self, hidden_states, st_mask, edges, cur_config):
        # st_mask: [1, max_seq_len * indoc_num + indoc_num + 1]
        hidden_states = hidden_states.reshape(1, -1, hidden_states.size(-1))
        max_token = cur_config['max_token']
        max_sentence = cur_config['max_sentence']
        sentence_start = cur_config['sentence_start']
        max_paragraph = cur_config['max_paragraph']
        st_mask = st_mask.view(1, -1)
        edges_src, edges_tgt, edges_type, edges_pos = edges
        graph_hidden = hidden_states.data.new(st_mask.size(0), st_mask.size(1), hidden_states.size(2)).fill_(0)
        graph_hidden[:, :max_token, :] = hidden_states
        mask = st_mask[:, sentence_start:].eq(1).unsqueeze(-1)
        # Update by TOKEN_TO_SENTENCE
        indices_t2s = edges_type.eq(EdgeType.TOKEN_TO_SENTENCE).nonzero().view(-1).tolist()
        graph_hidden = self.average_pooling(graph_hidden, edges_src[indices_t2s], edges_tgt[indices_t2s])
        para_start = sentence_start + max_sentence
        para_end = para_start + max_paragraph
        if 'para_vecs' in cur_config:
            para_end = para_start + cur_config['para_vecs'].size(0)
        graph_hidden[:, para_start: para_end, :] = graph_hidden[:, : sentence_start: self.max_indoc_token, :].detach() \
            if 'para_vecs' not in cur_config else cur_config['para_vecs'].detach()
        # Update by PARAGRAPH_TO_DOCUMENT
        indices_p2d = edges_type.eq(EdgeType.PARAGRAPH_TO_DOCUMENT).nonzero().view(-1).tolist()
        graph_hidden = self.average_pooling(graph_hidden, edges_src[indices_p2d], edges_tgt[indices_p2d])
        return graph_hidden

    @classmethod
    def average_pooling(cls, graph_hidden, edges_src, edges_tgt):
        batch_size, n_nodes, hidden_size = graph_hidden.size()
        graph_hidden = graph_hidden.view(batch_size * n_nodes, hidden_size)
        src_tensor = graph_hidden[edges_src]

        indices = edges_tgt.view(-1, 1).expand(-1, hidden_size)
        # indices: [tgt_node_num, hidden_size]
        sum_hidden = graph_hidden.clone().fill_(0)
        sum_hidden.scatter_add_(dim=0, index=indices, src=src_tensor)

        n_edges = graph_hidden.data.new(batch_size * n_nodes).fill_(0)
        n_edges.scatter_add_(dim=0, index=edges_tgt, src=torch.ones_like(edges_tgt).float())
        indices = n_edges.nonzero().view(-1)
        graph_hidden[indices] += sum_hidden[indices] / n_edges[indices].unsqueeze(-1)

        return graph_hidden.view(batch_size, n_nodes, hidden_size)
