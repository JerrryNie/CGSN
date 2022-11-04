"""rename same titles for different docs"""
import json
from tqdm import tqdm
from html import unescape
files = ['./data/HotpotQA-Doc/split/hotpotqa-train-qasper-all.json',
         './data/HotpotQA-Doc/split/hotpotqa-dev-qasper-part0.json',
         './data/HotpotQA-Doc/split/hotpotqa-test-qasper-part0.json']
names = ['train', 'dev', 'test']

for file, name in zip(files, names):
    max_same_title = 0
    print('read [{}]'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    titles = {}
    for doc_id, doc in tqdm(data.items(), desc='process [{}]'.format(name), total=len(data)):
        title = doc['title']
        unescaped_title = unescape(title)
        if unescaped_title in titles:
            titles[unescaped_title].append(doc_id)
            data[doc_id]['title'] = title + '_{}'.format(len(titles[unescaped_title]))
        else:
            titles[unescaped_title] = [doc_id]
        max_same_title = max(max_same_title, len(titles[unescaped_title]))

    output_file = './data/HotpotQA-Doc/split/hotpotqa-doc-{}.json'.format(name)
    print('Write to [{}]'.format(output_file))
    print('max_same_title: [{}]'.format(max_same_title))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
