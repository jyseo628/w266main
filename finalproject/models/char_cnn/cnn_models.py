import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Embedding, Convolution1D, MaxPooling1D, Flatten, Dense, Dropout)


# A params dict example
params = {
    'alphabet_size': 71,
    'embedding_size': 128,
    'input_length': 1014,
    'filters': [],
    'kernal_size': [],
    'pool_size': [],
    'fully_connected_dim': [],
    'dropout_rate': [],
    'loss': 'binary_crossentropy',
    'activation': 'sigmoid',
    'lr': 0.0001,
}


def text_to_padding(df, alphabet_index, max_input_size):
    X = []
    for _, row in df.iterrows():
        str2idx = np.zeros(max_input_size, dtype='int64')
        for i, letter in enumerate(row['text'].lower()):
            if i == max_input_size:
                break
            str2idx[i] = alphabet_index.get(letter, 0)
        X.append(str2idx)
    return np.array(X)


def get_model(params):
    model = Sequential()

    embedding_layer = Embedding(
        params['alphabet_size'],
        params['embedding_size'],
        input_length=params['input_length'])
    model.add(embedding_layer)

    for filters, kernal_size, pool_size in zip(
            params['filters'], params['kernal_size'], params['pool_size']):
        conv_layer = Convolution1D(filters, kernal_size, activation='relu')
        model.add(conv_layer)

        if pool_size:
            model.add(MaxPooling1D(pool_size))

    model.add(Flatten())

    for unit, dropout_rate in zip(
            params['fully_connected_dim'], params['dropout_rate']):
        model.add(Dense(unit, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation=params['activation']))
    model.compile(
        loss=params['loss'],
        optimizer=Adam(lr=params['lr']),
        metrics=['accuracy'])

    print(model.summary())
    return model
