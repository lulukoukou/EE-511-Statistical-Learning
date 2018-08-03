#!/usr/bin/env python
# -*- coding: <utf_8> -*-
import tensorflow as tf
import csv
import sys
from langdetect import detect
import os
# used for unseen words in training vocabularies
unk = None
# sentence start and end
sentence_start = '<S>'
sentence_end = '</S>'
# preprocessing
count = 0
test_id = []
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
with open('test_no_id.tsv','r',encoding="utf_8") as fd:
    rd = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in rd:
        test_id.append(x for x in row)
language = ['es', 'en', 'pt', 'fr', 'ca', 'de', 'eu', 'it', 'gl']

with open('test.ids.csv', 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    for item in test_id:
        item = list(item)[0]
        temp = []
        language_type = str(detect(item))
        if language_type in language:
            temp.append(language_type)
        else:
            temp.append('en')
        writer.writerow(temp)
