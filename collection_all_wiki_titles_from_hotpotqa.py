"""Collect Wikipedia titles from Hotpotqa and download corresponding
    html files in ./data/wiki for each title"""
from datasets import load_dataset
from tqdm import tqdm
import pickle
import os
import requests
from time import sleep
import multiprocessing

if not os.path.exists('title_set.pkl'):
    title_set = set()
    # title_to_url = {}
    allset = load_dataset('hotpot_qa', 'distractor')
    for name in allset:
        for item in tqdm(allset[name], desc='collect titles in [{}]'.format(name)):
            titles = item['context']['title']
            title_set.update(titles)
    print('Total number of titles: [{}]'.format(len(title_set)))
    with open('title_set.pkl', 'wb') as f:
        pickle.dump(title_set, f)
else:
    with open('title_set.pkl', 'rb') as f:
        title_set = pickle.load(f)


url_format = "https://en.wikipedia.org/api/rest_v1/page/html/{}"
headers={'user-agent':"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
         "accept": "text/html; charset=utf-8; profile=\"https://www.mediawiki.org/wiki/Specs/HTML/2.1.0\""}


def title_clean(title):
    title = title.replace('/', '%2F').replace('&quot;', "\"").replace('&amp;', '&')
    return title

if not os.path.exists('./data/wiki'):
    os.makedirs('./data/wiki')

new_title_set = set()
for title in tqdm(title_set, desc='filter titles...'):
    _title = title_clean(title)
    page_name = '_'.join(_title.split())
    if os.path.exists('./data/wiki/{}.html'.format(page_name)):
        continue
    new_title_set.add(title)

title_set = new_title_set
print('[{}] titles need to be downloaded...'.format(len(title_set)))

def download_by_title(thread_id, titles):
    complete_flag = True
    while True:
        for title in titles:
            title = title_clean(title)
            page_name = '_'.join(title.split())
            if os.path.exists('./data/wiki/{}.html'.format(page_name)):
                print('title [{}] exists, skip...'.format(page_name))
                continue
            url = url_format.format(page_name)
            try:
                response = requests.get(url, headers=headers)
            except Exception as e:
                print('Error [{}] occurred when download page name [{}]'.format(e, page_name))
                complete_flag = False
                continue
            html = response.text
            with open('./data/wiki/{}.html'.format(page_name), 'w', encoding='utf-8') as f:
                f.write(html)
        if complete_flag:
            break
        complete_flag = True
    print('thread [{}] finished!'.format(thread_id))


num_threads = 64
p = multiprocessing.Pool(num_threads)
examples = list(title_set)
print('example num: [{}]'.format(len(examples)))
pool = []
for i in range(num_threads):
    start_index = len(examples) // num_threads * i
    end_index = len(examples) // num_threads * (i + 1)
    if i == num_threads - 1:
        end_index = len(examples)
    pool.append(p.apply_async(download_by_title, args=(i, examples[start_index: end_index])))
p.close()
p.join()

for i, thread in enumerate(pool):
    cached_path_tmp = thread.get()