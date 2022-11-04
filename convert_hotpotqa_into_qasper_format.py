"""Convert hotpotqa dataset with full documents into qasper format."""
from datasets import load_dataset
import json
import pickle
import os
import random
from tqdm import tqdm
from collections import Counter
from html import unescape
random.seed(42)


def title_clean(title):
    title = title.replace('/', '%2F').replace('&quot;', "\"").replace('&amp;', '&')
    return title


def text_filter(text):
    # remove some noisy characters
    assert isinstance(text, str)
    new_text = ''
    for c in text:
        if ord(c) != 1462 and ord(c) != 65533:
            new_text += c
    return new_text


def load_article(title):
    text_file_format = './data/wiki_text/{}.txt'
    title = title_clean(title)
    page_name = '_'.join(title.split())
    text_file = text_file_format.format(page_name)
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    paras = text.split('\n')
    part_text = []
    # split and insert each section as well as its paragraphs into {'section_name': '', 'paragraphs': []}
    for p in paras:
        p = text_filter(p)
        if p.startswith('[SECTION NAME]'):
            part_text.append({'section_name': p.replace('[SECTION NAME]', ''),
                              'paragraphs': []})
        else:
            if len(part_text) == 0:
                part_text.append({'section_name': '',
                                'paragraphs': []})
            if len(p) > 0:
                part_text[-1]['paragraphs'].append(p)
    final_part_text = []
    for sec in part_text:
        if len(sec['paragraphs']) > 0:
            final_part_text.append(sec)
    return final_part_text, unescape(title)


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def match_by_f1(part_text, sent):
    """Find the best text in list part_text, which matches sent the best"""
    assert isinstance(sent, str)
    assert isinstance(part_text, list)
    assert isinstance(part_text[0], str)
    best_score = 0
    best_para = None
    sent = sent.strip()
    for p in part_text:
        # return if p is a sub-string of sent
        if sent in p:
            return p
    for p in part_text:
        f1, precision, recall = f1_score(p, sent)
        # otherwise, take the p with the highest recall value
        if recall > best_score:
            best_para = p
            best_score = recall
    assert best_para
    return best_para

all_set = load_dataset('hotpot_qa', 'distractor')
trainset = [item for item in all_set['train']]
testset = [item for item in all_set['validation']]
train_size = len(trainset)
test_size = len(testset)
print('total qas: [{}]'.format(train_size + test_size))
random.shuffle(trainset)
dev_size = test_size // 3
devset = testset[: dev_size]
testset = testset[dev_size:]
train_size = len(trainset)
dev_size = len(devset)
test_size = len(testset)
print('train size: [{}]'.format(train_size))
print('dev size: [{}]'.format(dev_size))
print('test size: [{}]'.format(test_size))

if not os.path.exists('./data/HotpotQA-Doc/split'):
    os.makedirs('./data/HotpotQA-Doc/split')

split_sets = {'dev': devset, 'train': trainset, 'test': testset}
partition_num = 3600
for name, _set in split_sets.items():
    partion = {}
    p_cnt = 0
    qa_cnt = 0
    for item in tqdm(_set):
        level = item['level']
        if level != 'hard':
            continue
        qa_cnt += 1
        qid = item['id']
        question = item['question']
        answer = item['answer']
        sf = item['supporting_facts']
        sf_titles = sf['title']
        sf_sent_ids = sf['sent_id']
        assert len(sf_titles) == len(sf_sent_ids)
        context = item['context']
        full_doc_titles = context['title']
        sentences = context['sentences']
        assert len(full_doc_titles) == len(sentences)
        new_full_doc_titles = []
        new_sentences = []
        title_filter = set(sf_titles)
        for title, sents in zip(full_doc_titles, sentences):
            if title in sf_titles:
                new_full_doc_titles.append(title)
                new_sentences.append(sents)
        full_doc_titles = new_full_doc_titles
        sentences = new_sentences
        evidence = []
        full_text = []
        titles = []
        for title, sents in zip(full_doc_titles, sentences):
            part_text, part_title = load_article(title)
            full_text.extend(part_text)
            titles.append(part_title)
            new_part_text = []
            for sec in part_text:
                new_part_text.append(sec['section_name'])
                new_part_text.extend(sec['paragraphs'])
            for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
                if sf_title == title:
                    try:
                        sf_sent_txt = sents[sf_sent_id]
                    except Exception as e:
                        continue
                    try:
                        best_sf_para = match_by_f1(new_part_text, sf_sent_txt)
                    except Exception as e:
                        continue
                    evidence.append(best_sf_para)
        if len(evidence) == 0:
            print('Skip [{}]'.format(qid))
            continue
        partion[qid] = {
            'title': ' '.join(titles),
            'abstract': '',
            'full_text': full_text,
            'qas': [
                {
                    'question': question,
                    'question_id': qid,
                    'answers': [
                        {
                            'answer': {
                                'extractive_spans': [answer],
                                'evidence': evidence
                            }
                        }
                    ]
                }
            ]
        }
        if len(partion) == partition_num and name == 'train':
            output_file = './data/HotpotQA-Doc/split/hotpotqa-{}-qasper-part{}.json'.format(name, p_cnt)
            print('Write into [{}]...'.format(output_file))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(partion, f, indent=4)
                partion = {}
                p_cnt += 1
    if len(partion) > 0:
        output_file = './data/HotpotQA-Doc/split/hotpotqa-{}-qasper-part{}.json'.format(name, p_cnt)
        print('Write into [{}]...'.format(output_file))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(partion, f, indent=4)
            partion = {}
            p_cnt += 1
    print('qa_cnt: [{}]'.format(qa_cnt))
