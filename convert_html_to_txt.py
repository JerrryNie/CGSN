"""convert hotpotqa html pages into txt and save in ./data/wiki_text"""
import json
import os
import pickle
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import multiprocessing

with open('title_set.pkl', 'rb') as f:
    title_set = pickle.load(f)
title_pattern = re.compile(r'h[0-9]+')


def title_clean(title):
    title = title.replace('/', '%2F').replace('&quot;', "\"").replace('&amp;', '&')
    return title


def is_title(tag):
    if tag.name is None:
        return False
    assert isinstance(tag.name, str), '{}: {}'.format(tag.name, type(tag.name))
    if title_pattern.match(tag.name):
        return True
    else:
        return False


def clean_text(txt):
    txt = txt.replace('\n', ' ').replace('\t', ' ')
    txt = ' '.join(txt.split())
    return txt

if not os.path.exists('./data/wiki'):
    os.makedirs('./data/wiki')
if not os.path.exists('./data/wiki_text'):
    os.makedirs('./data/wiki_text')

new_title_set = set()
for title in tqdm(title_set):
    title = title_clean(title)
    page_name = '_'.join(title.split())
    input_html = './data/wiki/{}.html'.format(page_name)
    output_txt = './data/wiki_text/{}.txt'.format(page_name)
    if os.path.exists(output_txt):
        continue
    new_title_set.add(title)

title_set = new_title_set
print('Need to process [{}] docs'.format(len(title_set)))

def extract_text(thread_i, titles):
    for title in titles:
        title = title_clean(title)
        page_name = '_'.join(title.split())
        input_html = './data/wiki/{}.html'.format(page_name)
        output_txt = './data/wiki_text/{}.txt'.format(page_name)
        if os.path.exists(output_txt):
            continue
        with open(input_html, 'r', encoding='utf-8') as f:
            html = f.read()
        paras = []
        secs = []
        soup = BeautifulSoup(html, 'lxml')
        section_tags = soup.find_all('section')
        for section_tag in section_tags:
            new_subsection = False
            for tag in section_tag.children:
                if is_title(tag):
                    while secs and secs[-1]['h_tag'] >= tag.name:
                        secs.pop()
                    secs.append({
                        'h_tag': tag.name,
                        'text': clean_text(tag.get_text())
                    })
                    para_text = '[SECTION NAME]'
                    for idx, sec in enumerate(secs):
                        if idx:
                            para_text += ' ::: '
                        para_text += sec['text']
                    paras.append(para_text)
                    new_subsection = True
                if tag.name is not None and tag.name == 'p':
                    if new_subsection is False:
                        paras.append('[SECTION NAME]')
                        new_subsection = True
                    para_text = clean_text(tag.get_text())
                    paras.append(para_text)
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paras))
    print('Thread [{}] finished!'.format(thread_i))


num_threads = 32
p = multiprocessing.Pool(num_threads)
examples = list(title_set)
print('example num: [{}]'.format(len(examples)))
pool = []
for i in range(num_threads):
    start_index = len(examples) // num_threads * i
    end_index = len(examples) // num_threads * (i + 1)
    if i == num_threads - 1:
        end_index = len(examples)
    pool.append(p.apply_async(extract_text, args=(i, examples[start_index: end_index])))
p.close()
p.join()

for i, thread in enumerate(pool):
    cached_path_tmp = thread.get()
