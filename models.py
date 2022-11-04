import torch
import collections
import torch.nn as nn
from transformers import (BertPreTrainedModel, AutoModel, AutoModelForSequenceClassification, AutoConfig,
                          AutoModelForSeq2SeqLM)
from graph_encoder import MetaNodePosition, MetaEncoderIndoc
import pickle
from time import sleep


class CGSNSciBert(torch.nn.Module):
    # indoc: [CLS] + query + [SEP] + para + [SEP]
    def __init__(self, model_name_or_path, my_config, bert_config=None):
        super(CGSNSciBert, self).__init__()
        self.config = my_config
        self.indoc_num = my_config.indoc_num
        MetaNodePosition(indoc_num=my_config.indoc_num, max_sentence=my_config.meta_sentence_num,
                         max_paragraph=my_config.meta_paragraph_num,
                         max_chunk=my_config.meta_doc_num)
        self.max_indoc_token = my_config.max_indoc_token_len
        self.max_indoc_sentence = my_config.max_indoc_sentence_len
        self.max_indoc_paragraph = my_config.max_indoc_paragraph_len
        self.max_token = my_config.max_indoc_token_len * my_config.indoc_num
        self.model_name_or_path = model_name_or_path
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.requires_grad = my_config.requires_grad
        if self.requires_grad in ["none", "all"]:
            for param in self.bert.parameters():
                param.requires_grad = self.requires_grad == "all"
        else:
            model_name_regexes = self.requires_grad.split(",")
            for name, param in self.bert.named_parameters():
                found = False
                for regex in model_name_regexes:
                    if regex in name:
                        found = True
                        break
                param.requires_grad = found

        self.para_dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.para_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        self.hidden_size = self.bert.config.hidden_size
        self.encoder = MetaEncoderIndoc(my_config)
        self.evidence_updater = nn.Sequential(collections.OrderedDict([
            ('evidence_updater_dense_0', torch.nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ('evidence_updater_sigmoid_0', torch.nn.Sigmoid())
        ]))
        self.para_mem_merge = nn.Sequential(collections.OrderedDict([
            ('para_mem_merge_dense_0', torch.nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ('para_mem_merge_tanh_0', torch.nn.Tanh())
        ]))
        self.dropout = nn.Dropout(my_config.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, st_mask, st_index,
                graph_attention_mask, meta_hiddens=None, evidence_hiddens=None, edges=None,
                is_paragraph_start=None, candidate_idx=None, label_idx=None, start_positions=None):
        # each batch contains a segment of Q-P pairs of a single document
        # graph_attention_mask: [1, (max_indoc_token + max_indoc_sentence + max_indoc_paragraph) * indoc_num + 1]
        # graph_attention_mask, st_mask and st_index have the same size
        assert input_ids.size(0) == 1
        input_ids = input_ids.view(-1, self.max_indoc_token)
        indoc_num = input_ids.size(0)
        max_token = indoc_num * self.max_indoc_token
        max_sentence = self.indoc_num * self.max_indoc_sentence
        max_paragraph = self.indoc_num * self.max_indoc_paragraph
        cur_config = {
            'max_token': max_token,
            'sentence_start': self.max_token,
            'max_sentence': max_sentence,
            'max_paragraph': max_paragraph
        }
        attention_mask = attention_mask.view(indoc_num, -1)
        token_type_ids = token_type_ids.view(indoc_num, -1)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs['pooler_output'].unsqueeze(1)
        if evidence_hiddens is not None:
            assert meta_hiddens is not None
            para_meta_hiddens = meta_hiddens[:, MetaNodePosition.MAX_SENTENCE:
                                             MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH, :]
            h_prime = self.para_mem_merge(evidence_hiddens)
            z = self.evidence_updater(evidence_hiddens)
            h_present = (1 - z) * para_meta_hiddens + z * h_prime
            meta_node_output = torch.cat([meta_hiddens[:, : MetaNodePosition.MAX_SENTENCE, :],
                                          h_present,
                                          meta_hiddens[:, MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH:,
                                                       :]
                                          ], dim=1)
            assert meta_node_output.size() == meta_hiddens.size()
            meta_hiddens = meta_node_output
        graph_output, meta_node_output = self.encoder(hidden_states=outputs[0],
                                                      meta_hiddens=meta_hiddens,
                                                      graph_attention_mask=graph_attention_mask,
                                                      st_mask=st_mask,
                                                      edges=edges, segment_ids=token_type_ids,
                                                      cur_config=cur_config,
                                                      output_all_encoded_layers=False)
        graph_output = graph_output[0]
        _meta_node_output = meta_node_output[0]

        # paragraph
        para_start = self.max_token + max_sentence
        para_end = para_start + max_paragraph
        para_output = torch.tanh(self.para_dense(graph_output[:, para_start:para_end, :]))
        para_logits = self.para_outputs(para_output).reshape(-1, 2)
        para_zero_mask = st_index[:, para_start: para_end].eq(-1).cuda()

        evidence_hiddens_output = None
        para_logits_one = para_logits.reshape(st_index.size(0), max_paragraph, -1)

        para_logits_one[:, :, 1] = torch.where(para_zero_mask, torch.full_like(para_logits_one[:, :, 1], float('-inf')),
                                                para_logits_one[:, :, 1])
        para_weights = torch.nn.functional.softmax(para_logits_one[:, :, 1], dim=-1).unsqueeze(-1)  # [batch_size, max_para_len, 1]
        para_weights = torch.transpose(para_weights, 1, 2)  # [batch_size, 1, max_para_len]
        para_weights_clean = torch.where(torch.isnan(para_weights), torch.full_like(para_weights, 0.), para_weights)
        evidence_summary = torch.matmul(para_weights_clean, graph_output[:, para_start:para_end, :])
        evidence_summary = evidence_summary.repeat(1, MetaNodePosition.MAX_PARAGRAPH, 1)
        para_meta_hiddens = _meta_node_output[:, MetaNodePosition.MAX_SENTENCE:
                                                MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH, :]
        para_mem_combined = torch.cat([para_meta_hiddens, evidence_summary], dim=-1)
        evidence_hiddens_output = para_mem_combined.detach()

        meta_node_output = _meta_node_output
        outputs = (para_logits.reshape(st_index.size(0), max_paragraph, -1),
                   st_index[:, para_start:para_end], meta_node_output.detach() if meta_node_output is not None else None,
                   evidence_hiddens_output)
        if start_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            _start_positions = torch.zeros_like(para_zero_mask).long()
            _start_positions = _start_positions.masked_fill(para_zero_mask, -1)
            for idx, start_position in enumerate(start_positions):
                if torch.any(start_position == -2).item():
                    continue
                _start_positions[idx, start_position] = 1
            assert para_logits.size(0) == len(_start_positions.view(-1))
            ## para_logits: [batch_size * para_num, 2]
            para_loss = loss_fct(para_logits, _start_positions.long().view(-1))
            total_loss = para_loss
            return (total_loss,) + outputs
        else:
            return outputs


class CGSNLEDEncoder(torch.nn.Module):
    # indoc: <s> + query + </s> + </s> + para + </s>
    def __init__(self, model_name_or_path, my_config, bert_config=None):
        super(CGSNLEDEncoder, self).__init__()
        self.config = my_config
        self.indoc_num = my_config.indoc_num
        MetaNodePosition(indoc_num=my_config.indoc_num, max_sentence=my_config.meta_sentence_num,
                         max_paragraph=my_config.meta_paragraph_num,
                         max_chunk=my_config.meta_doc_num)
        self.max_indoc_token = my_config.max_indoc_token_len
        self.max_indoc_sentence = my_config.max_indoc_sentence_len
        self.max_indoc_paragraph = my_config.max_indoc_paragraph_len
        self.max_token = my_config.max_indoc_token_len * my_config.indoc_num

        self.attention_dropout = 0.1
        self.attention_window_size = 4
        self.led_config = AutoConfig.from_pretrained(model_name_or_path)
        self.led_config.attention_dropout = self.attention_dropout
        self.led_config.attention_window = [self.attention_window_size] * len(self.led_config.attention_window)
        self.num_labels = 2
        self.led_config.num_labels = self.num_labels
        led_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.led_config)
        self.model_name_or_path = model_name_or_path
        self.led_encoder = led_model.led.encoder
        self.requires_grad = my_config.requires_grad
        if self.requires_grad in ["none", "all"]:
            for param in self.led_encoder.parameters():
                param.requires_grad = self.requires_grad == "all"
        else:
            model_name_regexes = self.requires_grad.split(",")
            for name, param in self.led_encoder.named_parameters():
                found = False
                for regex in model_name_regexes:
                    if regex in name.split('.'):
                        found = True
                        break
                param.requires_grad = found

        self.para_dense = nn.Linear(self.led_config.hidden_size, self.led_config.hidden_size)
        self.para_outputs = nn.Linear(self.led_config.hidden_size, 2)
        self.hidden_size = self.led_config.hidden_size
        self.encoder = MetaEncoderIndoc(my_config)
        self.evidence_updater = nn.Sequential(collections.OrderedDict([
            ('evidence_updater_dense_0', torch.nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ('evidence_updater_sigmoid_0', torch.nn.Sigmoid())
        ]))
        self.para_mem_merge = nn.Sequential(collections.OrderedDict([
            ('para_mem_merge_dense_0', torch.nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ('para_mem_merge_tanh_0', torch.nn.Tanh())
        ]))
        self.dropout = nn.Dropout(my_config.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, st_mask, st_index,
                graph_attention_mask, meta_hiddens=None, evidence_hiddens=None, edges=None,
                is_paragraph_start=None, candidate_idx=None, label_idx=None, start_positions=None):
        # each batch contains a segment of Q-P pairs of a single document
        # graph_attention_mask: [1, (max_indoc_token + max_indoc_sentence + max_indoc_paragraph) * indoc_num + 1]
        # graph_attention_mask, st_mask and st_index have the same size
        assert input_ids.size(0) == 1
        input_ids = input_ids.view(-1, self.max_indoc_token)
        indoc_num = input_ids.size(0)
        max_token = indoc_num * self.max_indoc_token
        max_sentence = self.indoc_num * self.max_indoc_sentence
        max_paragraph = self.indoc_num * self.max_indoc_paragraph
        cur_config = {
            'max_token': max_token,
            'sentence_start': self.max_token,
            'max_sentence': max_sentence,
            'max_paragraph': max_paragraph
        }
        attention_mask = attention_mask.view(indoc_num, -1)
        token_type_ids = token_type_ids.view(indoc_num, -1)
        eos_mask = input_ids.eq(self.led_config.eos_token_id)
        outputs = self.led_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   global_attention_mask=attention_mask)
        hiddens = outputs[0]
        para_vecs = hiddens[eos_mask, :].view(hiddens.size(0), -1, hiddens.size(-1))[
            :, -1, :
        ]
        cur_config['para_vecs'] = para_vecs
        if evidence_hiddens is not None:
            assert meta_hiddens is not None
            para_meta_hiddens = meta_hiddens[:, MetaNodePosition.MAX_SENTENCE:
                                             MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH, :]
            h_prime = self.para_mem_merge(evidence_hiddens)
            z = self.evidence_updater(evidence_hiddens)
            h_present = (1 - z) * para_meta_hiddens + z * h_prime
            meta_node_output = torch.cat([meta_hiddens[:, : MetaNodePosition.MAX_SENTENCE, :],
                                          h_present,
                                          meta_hiddens[:, MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH:,
                                                       :]
                                          ], dim=1)
            assert meta_node_output.size() == meta_hiddens.size()
            meta_hiddens = meta_node_output

        graph_output, meta_node_output = self.encoder(hidden_states=outputs[0],
                                                      meta_hiddens=meta_hiddens,
                                                      graph_attention_mask=graph_attention_mask,
                                                      st_mask=st_mask,
                                                      edges=edges, segment_ids=token_type_ids,
                                                      cur_config=cur_config,
                                                      output_all_encoded_layers=False)

        graph_output = graph_output[0]
        _meta_node_output = meta_node_output[0]
        # paragraph
        para_start = self.max_token + max_sentence
        para_end = para_start + max_paragraph
        para_output = torch.tanh(self.para_dense(graph_output[:, para_start:para_end, :]))
        para_logits = self.para_outputs(para_output).reshape(-1, 2)
        para_zero_mask = st_index[:, para_start: para_end].eq(-1).cuda()

        evidence_hiddens_output = None
        para_logits_one = para_logits.reshape(st_index.size(0), max_paragraph, -1)

        para_logits_one[:, :, 1] = torch.where(para_zero_mask, torch.full_like(para_logits_one[:, :, 1], float('-inf')),
                                            para_logits_one[:, :, 1])
        para_weights = torch.nn.functional.softmax(para_logits_one[:, :, 1], dim=-1).unsqueeze(-1)  # [batch_size, max_para_len, 1]
        para_weights = torch.transpose(para_weights, 1, 2)  # [batch_size, 1, max_para_len]
        para_weights_clean = torch.where(torch.isnan(para_weights), torch.full_like(para_weights, 0.), para_weights)
        evidence_summary = torch.matmul(para_weights_clean, graph_output[:, para_start:para_end, :])
        evidence_summary = evidence_summary.repeat(1, MetaNodePosition.MAX_PARAGRAPH, 1)
        para_meta_hiddens = _meta_node_output[:, MetaNodePosition.MAX_SENTENCE:
                                            MetaNodePosition.MAX_SENTENCE + MetaNodePosition.MAX_PARAGRAPH, :]
        para_mem_combined = torch.cat([para_meta_hiddens, evidence_summary], dim=-1)
        evidence_hiddens_output = para_mem_combined.detach()

        meta_node_output = _meta_node_output
        # training
        outputs = (para_logits.reshape(st_index.size(0), max_paragraph, -1),
                   st_index[:, para_start:para_end], meta_node_output.detach() if meta_node_output is not None else None,
                   evidence_hiddens_output)
        if start_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            _start_positions = torch.zeros_like(para_zero_mask).long()
            _start_positions = _start_positions.masked_fill(para_zero_mask, -1)
            for idx, start_position in enumerate(start_positions):
                if torch.any(start_position == -2).item():
                    continue
                _start_positions[idx, start_position] = 1
            assert para_logits.size(0) == len(_start_positions.view(-1))
            ## para_logits: [batch_size * para_num, 2]
            para_loss = loss_fct(para_logits, _start_positions.long().view(-1))
            total_loss = para_loss
            return (total_loss,) + outputs
        else:
            return outputs
