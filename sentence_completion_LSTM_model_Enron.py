# -*- coding: utf-8 -*-
""" Original file is located at
    https://colab.research.google.com/drive/1MxNI-AkojLzzwIvu_yCX4dQ6nqtBqtd8
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, CuDNNLSTM, Flatten, TimeDistributed, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, '')
    return text


def preprocess(data):
    output = []
    punct = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
    for line in data:
         pline= clean_special_chars(line.lower(), punct)
         output.append(pline)
    return output  


def generate_dataset():
    processed_corpus = preprocess(corpus)    
    output = []
    for line in processed_corpus:
        token_list = line
        for i in range(1, len(token_list)):
            data = []
            x_ngram = '<start> '+ token_list[:i+1] + ' <end>'
            y_ngram = '<start> '+ token_list[i+1:] + ' <end>'
            data.append(x_ngram)
            data.append(y_ngram)
            output.append(data)
    print("Dataset prepared with prefix and suffixes for teacher forcing technique")
    dummy_df = pd.DataFrame(output, columns=['input','output'])
    return output, dummy_df


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word


def max_length(t):
    return max(len(i) for i in t)


def load_data_portions():
    pairs, df = generate_dataset()
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_length(input_data), max_length(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")

    target_data = [[output_data[n][i+1] for i in range(len(output_data[n])-1)] for n in range(len(output_data))]
    target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
    target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

    p = np.random.permutation(len(input_data))

    return input_data[p], output_data[p], target_data[p], in_lang, out_lang, max_length_in, max_length_out, df


def get_data():
    # Enron https://data.world/brianray/enron-email-dataset
    data = pd.read_csv("enron_05_17_2015_with_labels_v2.csv")

    # Enron https://data.world/brianray/enron-email-dataset part wise
    # data = pd.read_csv("enron_05_17_2015_with_labels_v2_100k_chunk_1_of_6.csv", ignore_index=True))
    # data.append(pd.read_csv("enron_05_17_2015_with_labels_v2_100k_chunk_2_of_6.csv", ignore_index=True))

    # For content less than 100 words and that are Non-replies
    # data = data[(data['content'].str.len() <100) & ~(data['subject'].str.contains('Re:', na=False))]
    # to_csv('enron_sample_data_100NR.csv', index=False, encoding='utf8')

    return data['content']


def design_model():
    # Creating the Encoder layers.
    encoder_inputs = Input(shape=(len_input,))
    encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)

    # Bidirectional LSTM
    encoder_lstm = Bidirectional(CuDNNLSTM(units=units, return_sequences=True, return_state=True))
    encoder_out, fstate_h, fstate_c, bstate_h, bstate_c = encoder_lstm(encoder_emb(encoder_inputs))
    state_h = Concatenate()([fstate_h, bstate_h])
    state_c = Concatenate()([fstate_c, bstate_c])

    encoder_states = [state_h, state_c]

    # Creating the Decoder layers.
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
    decoder_lstm = CuDNNLSTM(units=units * 2, return_sequences=True, return_state=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)
    # Two dense layers are added to model to improve inference capabilities.
    decoder_d1 = Dense(units, activation="relu")
    decoder_d2 = Dense(vocab_out_size, activation="softmax")
    decoder_out = decoder_d2(Dropout(rate=.2)(decoder_d1(Dropout(rate=.2)(decoder_lstm_out))))

    return [encoder_inputs, decoder_inputs], decoder_out


if __name__ == "__main__"():
    corpus = get_data()

    input_data, teacher_data, target_data, input_lang, target_lang, len_input, len_target, df = load_data_portions()

    pd.set_option('display.max_colwidth', -1)
    BUFFER_SIZE = len(input_data)
    BATCH_SIZE = 128
    embedding_dim = 300
    units = 128
    vocab_in_size = len(input_lang.word2idx)
    vocab_out_size = len(target_lang.word2idx)
    df.iloc[140:150]

    # Model_inputs has encoder and decoder inputs combined.
    model_inputs, model_outputs = design_model()
    model = Model(inputs=model_inputs, outputs=model_outputs)

    # With 'sparse_categorical_crossentropy' need not expand 'decoder_out' into a massive one-hot array.
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    epochs = 5
    history = model.fit([input_data, teacher_data],
                        target_data,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=0.2)

    import pickle
    from datetime import date

    pickle.dump(model, open("smart_compose_model_" + date.today().strftime("%y%m%d") + "_1.pkl", 'wb'))

    # Plot the results of the training.
    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.show()
