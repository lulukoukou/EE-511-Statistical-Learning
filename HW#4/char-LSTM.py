import tensorflow as tf
import numpy as np
import csv


MAX_LENGTH = 160
BATCH_SIZE = 32

class DataSet(object):
    def __init__(self, path):
        self.path = path
        self.epoches_completed = 0
        self.index_in_epoch = 0

        self.char_ids = []
        self.lang_ids = []
        self.seq_length = []
        self.num_samples = 0


    def preprocessing(self):

        k = 10
        vocab = {'<S>': 0, '</S>': 0}

        with open('lang_id_data/train.tsv', encoding="utf8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            # quoting=csv.QUOTE_NONE otherwise it cannot separate correctly when encountered double quotes
            for row in reader:
                vocab['<S>'] += 1
                vocab['</S>'] += 1
                for char in row[1]:
                    if char in vocab.keys():
                        vocab[char] += 1
                    else:
                        vocab[char] = 1

        delete = []
        vocab_cnt = 0
        outofvocab_cnt = 0
        vocab_type = 0
        for key, value in vocab.items():
            if value >= k:
                vocab_cnt += value
                vocab_type += 1
            else:
                outofvocab_cnt += value
                delete.append(key)

        for key in delete:
            del vocab[key]
        vocab['oov'] = outofvocab_cnt



        n_char = 1 # save 0 for padding
        n_lang = 0
        char_map = {}
        for key in vocab.keys():
            char_map[key] = n_char
            n_char += 1
        lang_map = {}

        with open(self.path, encoding="utf8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
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

class LSTMRNN(object):
    def __init__(self, batch_size, n_steps, char_size, lang_size, char_embdsize, lang_embdsize, cell_size):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.char_embdsize = char_embdsize
        self.lang_embdsize = lang_embdsize
        self.input_size = char_embdsize + lang_embdsize
        self.cell_size = cell_size
        self.char_size = char_size
        self.lang_size = lang_size

        self.x = tf.zeros([self.batch_size, self.n_steps, self.input_size], dtype=tf.float32, name=None)

        with tf.name_scope('inputs'):
            self.x_char = tf.placeholder(tf.int32, shape=[None, self.n_steps])
            self.x_lang = tf.placeholder(tf.int32, shape=[None, self.n_steps])
            self.weight4loss = tf.placeholder(tf.int32, shape=[None, self.n_steps-1])
            self.y = tf.placeholder(tf.int32, shape=[None, self.n_steps-1])
        with tf.variable_scope('embedding'):
            self.embedding_layer()
        # with tf.variable_scope('in_hidden'):
        #     self.input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.recurrent_layer()
        with tf.variable_scope('out_hidden'):
            self.output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.cost)


    def embedding_layer(self):
        char_embedding = tf.get_variable("char_embedding", [self.char_size, self.char_embdsize])
        lang_embedding = tf.get_variable("lang_embedding", [self.lang_size, self.lang_embdsize])

        embedded_char_id_x = tf.nn.embedding_lookup(char_embedding, self.x_char)
        embedded_lang_id_x = tf.nn.embedding_lookup(lang_embedding, self.x_lang)
        self.x = tf.concat([embedded_char_id_x, embedded_lang_id_x], 2)
        # self.y = self.x[:, 1:, :]
        self.x = self.x[:, :-1, :]

    # def input_layer(self):
    #     l_in_x = tf.reshape(self.x, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
    #     # Ws (in_size, cell_size)
    #     Ws_in = self._weight_variable([self.input_size, self.cell_size])
    #     # bs (cell_size, )
    #     bs_in = self._bias_variable([self.cell_size,])
    #     # l_in_y = (batch * n_steps, cell_size)
    #     with tf.name_scope('Wx_plus_b'):
    #         l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
    #     # reshape l_in_y ==> (batch, n_steps, cell_size)
    #     self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps-1, self.cell_size], name='2_3D')

    def recurrent_layer(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.x, initial_state=self.cell_init_state, time_major=False)
        # print(self.cell_outputs.shape)

    def output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        initializer_w = tf.random_normal_initializer(mean=0., stddev=1., )
        Ws_out = tf.get_variable(name='w', shape=[self.cell_size, self.char_size], initializer=initializer_w)
        initializer_b = tf.constant_initializer(0.1)
        bs_out = tf.get_variable(name='b', shape=[self.char_size, ], initializer=initializer_b)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
            self.pred = tf.reshape(self.pred, [self.batch_size, self.n_steps-1, self.char_size], name='2_3D')

    def compute_cost(self):
        print(self.pred.shape)
        print(self.y.shape)
        print(self.weight4loss)
        self.cost = tf.contrib.seq2seq.sequence_loss(
                logits=self.pred,
                targets=self.y,
                weights=self.weight4loss,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None,
                name=None)

if __name__ == '__main__':
    train_data = DataSet('lang_id_data/train.tsv')
    char_size, lang_size = train_data.preprocessing()

    model = LSTMRNN(batch_size=BATCH_SIZE, n_steps=MAX_LENGTH, char_size=char_size,
                    lang_size=lang_size, char_embdsize=15, lang_embdsize=6, cell_size=1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 0

    for i in range(10000):
        x_c, x_l, x_s, y_c = train_data.next_batch(BATCH_SIZE)
        if train_data.epoches_completed > epoch:
            epoch = train_data.epoches_completed
            print(epoch)
        if i == 0:
            feed_dict = {
                model.x_char: x_c,
                model.x_lang: x_l,
                model.weight4loss: x_s,
                model.y: y_c
                # create initial state
            }
        else:
            feed_dict = {
                model.x_char: x_c,
                model.x_lang: x_l,
                model.weight4loss: x_s,
                model.y: y_c,
                model.cell_init_state: state  # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        if i % 100 == 0:
            print('cost: ', round(cost, 4))
