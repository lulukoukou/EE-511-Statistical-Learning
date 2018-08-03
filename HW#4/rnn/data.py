import csv

class data(object)
    def __init__(self, filename):

    def preprocess_data(self):
        # used for unseen words in training vocabularies
        unk = 'unk'
        # sentence start and end
        sentence_start = '<S>'
        sentence_end = '</S>'
        # vocabulary dictionary
        voca = {}
        # read train.tsv file
        with open('lang_id_data/train.tsv', encoding="utf8") as read_file:
            row_reader = csv.reader(read_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            voca[sentence_start] = 0
            voca[sentence_end] = 0
            for row in row_reader:
                voca[sentence_start] += 1
                voca[sentence_end] += 1
                for item in list(row[1]):
                    if item in vocab.keys():
                        voca[item] += 1
                    else:
                        voca[item] = 1
        voca_size = 0
        out_of_voca_size = 0
        out_of_voca = {}
        for key in voca:
            if voca[key] >= 10:
                voca_size += 1
            else:
                out_of_voca [key] = voca[key]
                out_of_voca_size += 1
            
