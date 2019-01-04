import numpy as np
import matplotlib.pyplot as plt
import zipfile

from keras.models import Sequential
from keras.layers import (
    Dense, Embedding, CuDNNLSTM, CuDNNGRU, SpatialDropout1D, LSTM,
    GRU, Bidirectional)


# A params dict example
params = {
    'model_type': CuDNNLSTM,
    'batch_size': 256,
    'embedding_type': 'randomly initialized',
    'embedding_dim': 200,
    'embedding_max_features': 10000,
    'embedding_word_index': None,
    'embedding_pretrained_zip_path': None,
    'embedding_pretrained_name': None,
    'epoch': 6,
    'rnn_dim': 100,
    'rnn_layer_num': 1,
    'input_length': 40,
    'input_dropout': 0.5,
    'recurrent_dropout': 0.5,
    'optimizer': 'adam',
    'activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'is_bidirectional': False,
}


def read_zip_file(filepath, filename):
    zfile = zipfile.ZipFile(filepath)
    ifile = zfile.open(filename)
    for line in ifile.readlines():
        yield line


def get_model(params):
    model = Sequential()

    if params['embedding_type'] == 'randomly initialized':
        embedding_layer = Embedding(
            params['embedding_max_features'],
            params['embedding_dim'],
            input_length=params['input_length'])
    else:
        embedding_layer = get_pretrained_embedding_layer(
            params['embedding_pretrained_zip_path'],
            params['embedding_pretrained_name'],
            params['embedding_word_index'],
            params['embedding_dim'],
            params['input_length'],
            True)
    model.add(embedding_layer)

    # Given shape(x) = [samples, timesteps, channels], it uses noise_shape =
    # [samples, 1, channels] and drops entire 1-D feature maps. Dropout matrix
    # with this shape applies the same dropout rate at all timesteps.
    if params['input_dropout'] is not None:
        model.add(SpatialDropout1D(rate=params['input_dropout']))

    for rnn_layer in get_rnn_layers(params):
        if params.get('is_bidirectional'):
            model.add(Bidirectional(rnn_layer))
        else:
            model.add(rnn_layer)

    model.add(Dense(2, activation=params['activation']))
    model.compile(
        loss=params['loss'],
        optimizer=params['optimizer'],
        metrics=['accuracy'])

    print(model.summary())
    return model


def get_embeddings_index(zip_embedding_path, embedding_name):
    embeddings_index = {}
    for line in read_zip_file(zip_embedding_path, embedding_name):
        values = line.decode().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(embeddings_index, word_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_pretrained_embedding_layer(
        zip_embedding_path, embedding_name, word_index, embedding_dim,
        input_length, trainable):
    embeddings_index = get_embeddings_index(zip_embedding_path, embedding_name)
    embedding_matrix = get_embedding_matrix(
        embeddings_index, word_index, embedding_dim)
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=trainable)
    return embedding_layer


def get_rnn_layers(params):
    rnn_type = params['model_type']

    if rnn_type == LSTM or rnn_type == GRU:
        for _ in range(params['rnn_layer_num'] - 1):
            yield rnn_type(
                params['rnn_dim'],
                recurrent_dropout=params['recurrent_dropout'],
                return_sequences=True)

        yield rnn_type(
            params['rnn_dim'], recurrent_dropout=params['recurrent_dropout'])
    elif rnn_type == CuDNNLSTM or rnn_type == CuDNNGRU:
        for _ in range(params['rnn_layer_num'] - 1):
            yield rnn_type(params['rnn_dim'], return_sequences=True)

        yield rnn_type(params['rnn_dim'])


def plot_model_train_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.xlim(left=1)
    plt.show()
