"""Regard retreived evidence as the evidence in the dataset for question answering"""
import json
import sys
import os
data = json.load(open(sys.argv[1]))
result_path = sys.argv[2]
file_name = result_path.split('/')[-2]
model_name = sys.argv[3]
file_name += '_' + model_name
threshold = float(sys.argv[4])
mode = sys.argv[5]


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
                print('Warning: find nonstr, convert to empty str')
                cleaned_txts.append("")
        return cleaned_txts
    else:
        print('Warning: find nonstr, convert to empty str')
        return ""


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
print('cnt_predict: {}'.format(cnt_predict))
print('cnt_true: {}'.format(cnt_true))
print(labels_cnt)

max_retreived_evidence_num = 0
empty_evidence_cnt = 0
for _, paper_data in data.items():
    paragraphs = []
    for section_info in paper_data["full_text"]:
        paragraphs.append(
            text_clean(section_info['section_name'])
        )
        paragraphs.extend(text_clean(section_info["paragraphs"]))

    tokenized_corpus = [doc.split() for doc in paragraphs]

    for qa_info in paper_data["qas"]:
        predicted_labels = cross_paragraphs_real[qa_cnt]
        assert len(paragraphs) == len(predicted_labels)
        evidence = [paragraphs[i] for i, label in enumerate(predicted_labels) if label]
        if not evidence:
            empty_evidence_cnt += 1
            pass
        max_retreived_evidence_num = max(len(evidence),
                                         max_retreived_evidence_num)
        qid = qa_info['question_id']
        question = qa_info["question"]
        evidence_all = []
        label_all = []
        for answer_info in qa_info['answers']:
            answer_info['answer']['evidence'] = evidence
        qa_cnt += 1

output_dir = './data/Qasper'
with open(os.path.join(output_dir,
                       'qasper-{}-v0.2-predicted-evidence-{}-{}-{}.json'.format(mode, model_name, file_name, threshold)),
          'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
print(max_retreived_evidence_num)
print(empty_evidence_cnt)
print('Write to file: {}'.format(os.path.join(output_dir,
                                              'qasper-{}-v0.2-predicted-evidence-{}-{}-{}.json'.format(mode, model_name, file_name, threshold))))
