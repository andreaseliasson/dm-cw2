import os
import re
from bs4 import BeautifulSoup as bs
import pandas as pd


def get_sub_dir(root_dir):
    sub_dirs = []
    for root, dir_names, file_names in os.walk(root_dir):
        if dir_names:
            for name in dir_names:
                sub_dirs.append(name)
    return sub_dirs


def get_html_files_from_sub_dirs(root_dir, sub_dirs):
    text_docs = []
    for sub_dir in sub_dirs:
        for root, dir_names, file_names in os.walk(root_dir + sub_dir):
            text_docs.append((sub_dir, [file_name for file_name in file_names if re.search('html', file_name)]))
    return text_docs


def parse_html_files(text_docs):
    pattern = '^[A-Za-z.,:;!?()]+$'
    raw_docs = []
    for doc in text_docs:
        html_text = ''
        for html_file in doc[1]:
            html_file_path = '../data/gap-html/' + doc[0] + '/' + html_file
            html_soup = bs(open(html_file_path), 'html.parser')
            html_text_arr = ["".join(s.findAll(text=True)) for s in html_soup.findAll('span', {'class': 'ocr_cinfo'})]
            print(len(html_text_arr))
            html_text += " ".join([word for word in html_text_arr if re.match(pattern, word)]) + " "
            print(doc[0])
            print(html_file)
        raw_docs.append((doc[0], html_text))
    return raw_docs


def to_csv(docs):
    pd_frame = pd.DataFrame(docs, columns=['id', 'raw_text'])
    pd_frame.to_csv('../data/raw_text.csv', index=False)
