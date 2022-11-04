"""Converge all train parts into a single file"""
import json
import glob
import os
from tqdm import tqdm
input_file_format = './data/HotpotQA-Doc/split/hotpotqa-train-qasper-part[0-9]*.json'
input_files = glob.glob(input_file_format, recursive=True)
output_file = './data/HotpotQA-Doc/split/hotpotqa-train-qasper-all.json'

output_data = {}
for input_file in tqdm(input_files):
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        print('qas: [{}]'.format(len(input_data)))
    output_data.update(input_data)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)
    print('final qas: [{}]'.format(len(output_data)))
