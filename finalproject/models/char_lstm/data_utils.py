# coding=utf-8

# Dataset needs to be shuffled before calling TextReader() instance;
# Use shuffle_dataset() to make a new CSV file, default is /datasets/train_set.csv

from importlib import reload
import codecs
import random, csv
import numpy as np
from nltk.tokenize import word_tokenize
import string
import os
import unicodedata

# os.environ["PYTHONIOENCODING"] = "utf-8"
# import unidecode

printable = string.printable

# PATH needs to be changed accordingly
PATH = './'

SOURCE_TRAIN_SET = PATH + './../../data/training.1600000.processed.noemoticon.csv'
SOURCE_TEST_SET = PATH + './../../data/testdata.manual.2009.06.14.csv'
VALID_PERC = 0.05

# TODO: Add non-Ascii characters
emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '


DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
ALPHABET_SIZE = len(emb_alphabet)


def reshape_lines(lines):
    data = []
    for l in lines:
        split = l.decode('utf-8').split('\",\"')
        data.append((split[0][1:], split[-1][:-2]))
    print(data[:100])
    return data


def save_csv(out_file, data):
    with open(out_file, "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('Data saved to file: %s' % out_file)

    
def shuffle_datasets(valid_perc=VALID_PERC):
    ''' Shuffle the dataset '''
    assert os.path.exists(SOURCE_TRAIN_SET), 'Download the training set at http://help.sentiment140.com/for-students/'
    assert os.path.exists(SOURCE_TEST_SET), 'Download the testing set at http://help.sentiment140.com/for-students/'

    # Create training and validation set
    print('Creating training & validation set...')

    with codecs.open(SOURCE_TRAIN_SET, 'r', 'latin-1') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = [l.encode('utf-8') for l in lines]
        lines_train = lines[:int(len(lines) * (1 - valid_perc))]
        lines_valid = lines[int(len(lines) * (1 - valid_perc)):]

    save_csv(PATH + './../../data/valid_set.csv', reshape_lines(lines_valid))
    save_csv(PATH + './../../data/train_set.csv', reshape_lines(lines_train))

    print('Creating testing set...')

    with codecs.open(SOURCE_TEST_SET, 'r', 'latin-1') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = [l.encode('utf-8') for l in lines]
    save_csv(PATH + 'datasets/test_set.csv', reshape_lines(lines))
    print('All datasets have been created!')


class TextReader(object):
    """ Util for Reading the Stanford CSV Files """
    
    TRAIN_SET = './../../data/fully_cleansed_train_data.csv'
    TEST_SET = './../../data/fully_cleansed_test_data.csv'
    DEV_SET = './../../data/fully_cleansed_dev_data.csv'
    
    # Includes a range of long / short sentences and some junk ( that must get filtered out )
    QA_SET = './../data/fully_cleansed_qa_data.csv'
    
    def __init__(self, file, max_word_length, debug=False):
        # TextReader() takes a CSV file as input that it will read
        # through a buffer

        if file != None:
            self.file = file
        
        self.debug = debug
        self.max_word_length = max_word_length
    
    
    def encode_sentence(self,sentence):
        # Handle unknown characters to ' ' that are outside the embedding alphabet 
        encoded_sentence = ''
        for c in sentence.lower():
            if c in emb_alphabet:
                encoded_sentence += c
            else: 
                encoded_sentence += ' ' 
        return encoded_sentence
    
    
    def encode_one_hot(self, sentence):
        # Convert Sentences to np.array of Shape ('sentence_length', 'word_length', 'emb_size')

        max_word_length = self.max_word_length
        sent = []
        SENT_LENGTH = 0
        
        encoded_sentence = self.encode_sentence(sentence)
        
        # Handle if no content in the message (i.e. all removed) and force it to be 1 char long ('.')
        if len(encoded_sentence.strip())==0:
            encoded_sentence = '.'
                
        for word in word_tokenize(encoded_sentence):

            word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))

            for i, char in enumerate(word):

                try:
                    char_encoding = DICT[char]
                    one_hot = np.zeros(ALPHABET_SIZE)
                    one_hot[char_encoding] = 1
                    word_encoding[i] = one_hot

                except Exception as e:
                    pass

            sent.append(np.array(word_encoding))
            SENT_LENGTH += 1
        
        return np.array(sent), SENT_LENGTH

    
    def numpy_fillna(self, data, max_word_length):

        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(shape=(mask.shape + (max_word_length, ALPHABET_SIZE)), dtype='float32')
        out[mask] = np.concatenate(data)

        return out  
    
    
    def make_minibatch(self, sentences):
        
        # Create a minibatch of sentences and convert sentiment
        # to a one-hot vector, also takes care of padding

        max_word_length = self.max_word_length
        minibatch_x = []
        minibatch_y = []
        max_length = 0

        # data is a np.array of shape ('b', 's', 'w', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get ('b', 'SENTENCE_MAX_LENGTH', 'WORD_MAX_LENGTH', 'e')
                              
        for sentence in sentences:
                        
            # 0: Negative 1: Positive
            if self.debug:
                print('Original: ',sentence, '\nSentiment: ', sentence[-1],'\nSentence: ', sentence[:-2])
            
            # Extract sentiment +ve or -ve
            sentiment = np.array([0, 1]) if sentence[-1] == '0' else np.array([1, 0])
            minibatch_y.append(sentiment)
                            
            # One hot encode the characters in the sentence
            one_hot, length = self.encode_one_hot(sentence[:-2])
            
            if length >= max_length:
                max_length = length
            
            if self.debug:
                print('one_hot shape:', one_hot.shape)
            
            minibatch_x.append(one_hot)            
            
        # Padding...
        minibatch_x = self.numpy_fillna(minibatch_x, max_word_length)

        return minibatch_x, np.array(minibatch_y)

    
    def load_to_ram(self, batch_size):
        # Load n Rows from File f to Ram

        self.data = []
        n_rows = batch_size
        while n_rows > 0:
            file_line = next(self.file)
            # remove any trailing/leading spaces
            file_line = file_line.strip()
            # Check line isnt blank, if so skip it
            if (len(file_line)>0):
                self.data.append(file_line)
                n_rows -= 1
        if n_rows == 0:
            return True
        else:
            return False
        
        
    def _get_number_of_samples(self, dataset):
        
        n_samples = 0
                
        # Returns Next Batch and Catch Bound Errors
        if dataset == TextReader.TRAIN_SET:
            n_samples = 1024000
        elif dataset == TextReader.DEV_SET:
            n_samples = 256000
        elif dataset == TextReader.TEST_SET:
            n_samples = 320000
        elif dataset == TextReader.QA_SET:
            n_samples = 20
            
        if n_samples == 0:
            raise Exception('dataset type unknown must be from the enumeration TextReader.TRAIN/TEST/DEV/QA_SET')
        
        return n_samples
    
    
    def iterate_minibatch(self, batch_size, dataset, has_header=True):
        # Iterator for use in loop, to manage memory footprint
                
        n_samples = self._get_number_of_samples(dataset)
        
        # Trim header from file ( and track )
        if (has_header):
            self.header = next(self.file)
        
        # Calc batch size
        n_batch = int(n_samples // batch_size)
        
        # Iterate over batches, loading into RAM and making mini-batches
        for i in range(n_batch):
            if self.load_to_ram(batch_size):                
                inputs, targets = self.make_minibatch(self.data)
                yield inputs, targets

                
    def check_file(self, dataset, has_header=True):
        # Check file object for correct formating ('sentence',sentiment) and display line counts
        
        n=0
        eof=False        
        n_samples = self._get_number_of_samples(dataset)
                
        while (eof==False):
            try:
                file_line = next(self.file)
                file_line = file_line.strip()
                n+=1
                                    
                if  ((has_header and n>1) or not has_header):
                    # Check line isnt blank, if so skip it
                    if (len(file_line)==0):
                        print('Empty line @ ',n)
                    else:
                        try:
                            number = int(file_line[-1])
                        except (ValueError, IndexError):
                            number = None
                            if (number==None):
                                print('Bad line @',n, file_line[-1])
                            if (n%100000==0):
                                print('line @', n)
                else:
                    self.header=file_line

                if (n>=n_samples):
                    eof=True
                    
            except StopIteration:
                eof=True
        
        print('eof line count @', n)