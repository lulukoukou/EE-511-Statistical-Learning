#!/usr/bin/env python
# -*- coding: <utf_8> -*-
import tensorflow as tf
import csv
import sys
import math

# used for unseen words in training vocabularies
unk = 'unk'
# sentence start and end
sentence_start = '<s>'
sentence_end = '</s>'
count = 0
voca = {}
# read the train file
with open("train.tsv",encoding="utf_8") as fd:
    rd = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in rd:
        count += 1
        row = [x for x in row]
        temp = list(row[1])
        keys = voca.keys()
        for key in temp:
            if key in keys:
                voca[key] += 1
            else:
                voca[key] = 1
voca_list = {}
out_of_voca_list = {}
# appearing time>=10 is in the vocabulary
for key in voca:
    if voca[key] >= 10:
        voca_list[key] = voca[key]
    else:
        out_of_voca_list[key] = voca[key]
# include <S> and </S>
voca_list[sentence_start] = count
voca_list[sentence_end] = count
voca_list[unk] = len(out_of_voca_list)
# size of vocabulary and the percentage of out of vocabulary tokens
size = len(voca_list)
print ("The size of vocabulary is %s" %size)
percentage = 100*float(sum(out_of_voca_list.values()))/float(sum(voca.values()))
print ("The percentage of out_of_dictionary is %s" %percentage +"%")
# calcualte perplexity of valid data based on unigram model
valid_list = []
valid_dict = {}
valid_dict[unk] = voca_list[unk]
with open("val.tsv",encoding="utf_8") as fd:
    rd = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in rd:
        row = [x for x in row]
        temp = list(row[1])
        valid_list = list(set(valid_list+temp))
for item in valid_list:
    if item in voca_list.keys():
        valid_dict[item] = voca_list[item]

valid_dict[sentence_end] = voca_list[sentence_end]
denominator = sum(voca_list.values())
entropy = 0
# cross entropy equation and perplexity
for key in valid_dict:
    p = float(valid_dict[key])/float(denominator)
    entropy = entropy - (p)*math.log2(p)
print ("The entropy for iid is %s" %entropy)
perplexity = math.pow(2,entropy)
print ("The perplexity for iid is %s" %perplexity)
