import csv

class data(object)
    def __init__(self, path):
        self.path = path
        self.epoches_completed = 0
        self.index_in_epoch = 0

        self.char_ids = []
        self.lang_ids = []
        self.seq_length = []
        self.num_samples = 0


    def preprocessing(self):
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
            voca [sentence_start] = 0
            voca [sentence_end] = 0
            for row in row_reader:
                vocab[sentence_start] += 1
                vocab[sentence_end] += 1
                for item in row[1]:
                    if item in vocab.keys():
                        vocab[item] += 1
                    else:
                        vocab[item] = 1

        delete = []
        vocab_cnt = 0
        outofvocab_cnt = 0
        vocab_type = 0
        for key, value in vocab.items():
            if value >= 10:
                vocab_cnt += value
                vocab_type += 1
            else:
                outofvocab_cnt += value
                delete.append(key)

        for key in delete:
            del vocab[key]
        vocab['oov'] = outofvocab_cnt



        n_char = 1
        n_lang = 0
        char_map = {}
        for key in vocab.keys():
            char_map[key] = n_char
            n_char += 1
        lang_map = {}

        with open(self.path, encoding="utf8") as read_file:
            row_reader = csv.reader(read_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in row_reader:
                if line[0] not in lang_map:
                    lang_map[line[0]] = n_lang
                    n_lang += 1
                self.lang_ids.append([lang_map[line[0]]]*MAX_LENGTH)

                n_padding = MAX_LENGTH - len(line[1]) - 2
                self.seq_length.append([1]*(len(line[1])+2) + [0]*(n_padding-1))
                seq = [char_map['<S>']]
                for char in line[1]:
                    if char in warmup.delete:
                        seq.append(char_map['oov'])
                    else:
                        seq.append(char_map[char])
                seq.append(char_map['</S>'])
                seq += [0] * n_padding
                self.char_ids.append(seq)
        self.char_ids = np.array(self.char_ids)
        self.lang_ids = np.array(self.lang_ids)
        self.seq_length = np.array(self.seq_length)
        self.num_samples = self.char_ids.shape[0]
        # print(self.num_samples)
        return n_char, n_lang

    @property
    def char_ids_(self):
        return self.char_ids

    @property
    def lang_ids_(self):
        return self.lang_ids

    def seq_length_(self):
        return self.seq_length

    def next_batch(self, batch_size, shuffle=True):
        """Return the next batch_size sized samples."""
        start = self.index_in_epoch
        if start == 0 and shuffle:
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)
            self.char_ids = self.char_ids_[perm]
            self.lang_ids = self.lang_ids_[perm]
            self.seq_length = self.seq_length_[perm]
        # Extracts the next batch data.
        if start + batch_size > self.num_samples:
            self.epoches_completed += 1
            self.index_in_epoch = 0
            start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch += batch_size
        x_c = np.array(self.char_ids[start:end])
        x_l = np.array(self.lang_ids[start:end])
        x_s = np.array(self.seq_length[start:end])
        y_c = np.array(self.char_ids[start+1:end])
        return x_c, x_l, x_s
