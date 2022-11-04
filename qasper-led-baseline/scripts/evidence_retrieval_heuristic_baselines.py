import json
import sys
import random
import numpy as np
from typing import List, Tuple, Mapping, Dict
import glob

random.seed(31045)
beta = 1


def text_clean(text):
    if isinstance(text, str):
        cleaned_txt = text.replace("\n", " ").strip()
        cleaned_txt = ' '.join(cleaned_txt.split())
        cleaned_txt = cleaned_txt.strip()
        return cleaned_txt
    elif isinstance(text, list):
        cleaned_txts = []
        for txt in text:
            if isinstance(txt, str):
                cleaned_txt = txt.replace("\n", " ").strip()
                cleaned_txt = ' '.join(cleaned_txt.split())
                cleaned_txt = cleaned_txt.strip()
                cleaned_txts.append(cleaned_txt)
            else:
                cleaned_txts.append("")
        return cleaned_txts
    else:
        return ""


def compute_paragraph_f1_real(predicted_labels: List[int],
                              gold_labels: List[List[int]],
                              tag="") -> Tuple[List[int], List[int], List[int]]:
    f1s = []
    precisions = []
    recalls = []
    for gold_label in gold_labels:
        true_positives = sum([i and j for i, j in zip(predicted_labels, gold_label)])
        if sum(predicted_labels) == 0:
            precision = 1.0 if sum(gold_label) == 0 else 0.0
        else:
            precision = true_positives / sum(predicted_labels)
        recall = true_positives / sum(gold_label) if sum(gold_label) != 0 else 1.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append((1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall))
        precisions.append(precision)
        recalls.append(recall)
    if tag == 'cross' and max(f1s) > 0:
        global zeros
        zeros += 1
    return max(f1s), max(precisions), max(recalls)


data = json.load(open(sys.argv[1]))
threshold = float(sys.argv[3])
result_path_pattern = sys.argv[2]
result_paths = glob.glob(result_path_pattern)
if len(result_paths) > 1:
    result_paths = sorted(result_paths, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1])) if len(result_paths) > 1 else result_paths
for result_path in result_paths:
    print('result_path: {}'.format(result_path))
    qa_cnt = 0
    with open(result_path, 'r', encoding='utf-8') as f:
        cross_result = f.readlines()
    cross_results = {}
    cross_paragraphs_real = []
    for line in cross_result:
        fields = line.strip().split('\t')
        if len(fields) == 3:
            query_id = int(fields[0])
            para_id = int(fields[1])
            cross_label = int(float(fields[2]) > threshold)
        else:
            raise Exception('fields: {}'.format(fields))
        if query_id not in cross_results:
            cross_results[query_id] = {}
        cross_results[query_id][para_id] = cross_label

    cnt_predict = 0
    cnt_true = 0
    labels_cnt = {}
    for query_idx in range(len(cross_results)):
        labels = []
        label_dict = cross_results[query_idx]
        for para_idx in range(len(label_dict)):
            cnt_predict += 1
            labels.append(label_dict[para_idx])
            if labels[-1] == 1:
                cnt_true += 1
            if labels[-1] not in labels_cnt:
                labels_cnt[labels[-1]] = 0
            labels_cnt[labels[-1]] += 1
        cross_paragraphs_real.append(labels)
    zeros = 0
    gold_evidence = []

    gold_label = []
    all_paragraphs = []
    qa_cnt = 0
    for _, paper_data in data.items():
        paragraphs = []
        for section_info in paper_data["full_text"]:
            paragraphs.append(
                text_clean(section_info['section_name'])
            )
            paragraphs.extend(text_clean(section_info["paragraphs"]))
        if not paragraphs:
            paragraphs = [""]

        tokenized_corpus = [doc.split() for doc in paragraphs]

        for qa_info in paper_data["qas"]:
            qa_cnt += 1
            qid = qa_info['question_id']
            question = qa_info["question"]
            evidence_all = []
            label_all = []
            for answer_info in qa_info['answers']:
                evidences = []
                labels = [0] * len(paragraphs)
                for evidence in text_clean(answer_info["answer"]["evidence"]):
                    if 'FLOAT SELECTED' not in evidence:
                        evidences.append(evidence)
                        labels[paragraphs.index(evidence)] = 1
                evidence_all.append(evidences)
                label_all.append(labels)
            gold_evidence.append(evidence_all)
            all_paragraphs.append(paragraphs)
            gold_label.append(label_all)

    assert len(gold_label) == len(all_paragraphs)
    assert len(gold_label) == len(cross_paragraphs_real), '{}: {}'.format(
        len(gold_label), len(cross_paragraphs_real)
    )

    uppperbound_f1s = []
    uppperbound_f1s_real = []
    cross_f1s_real = []
    cross_precisions_real = []
    cross_recalls_real = []

    for idx, (cross_paragraph, evidence, paragraphs) in enumerate(zip(
            cross_paragraphs_real,
            gold_label,
            all_paragraphs)):
        cross_f1, cross_precision, cross_recall = compute_paragraph_f1_real(cross_paragraph, evidence, tag='cross')
        cross_f1s_real.append(cross_f1)
        cross_precisions_real.append(cross_precision)
        cross_recalls_real.append(cross_recall)
        uppperbound_f1s_real.append(compute_paragraph_f1_real(evidence[-1], evidence)[0])

    print('cur threshold: {}'.format(threshold))
    print('cur f1: {}'.format(np.mean(cross_f1s_real)))
    print('cur recall: {}'.format(np.mean(cross_recalls_real)))
    print('cur precision: {}'.format(np.mean(cross_precisions_real)))
