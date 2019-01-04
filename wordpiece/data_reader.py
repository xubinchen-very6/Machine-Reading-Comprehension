from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import json
from tqdm import tqdm


class Vocab_token:
    
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []
    '''token是否在vocab内，返回该token的index
       如果不在把这个词加入vocab'''
    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)
    
class Vocab_stroke:

    def __init__(self, token2index=None, index2token=None):
        
        with open('../data/util_data/wordpiece2index.json') as f:
            wordpiece2index = json.load(f)
        self._token2index = wordpiece2index     #wordpiece2index
        self._index2token = index2token or []   #index2wordpiece
        self.word_stroke_dict = self.w2s()      #word2wordpiece
        
    def is_Chinese(self,word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    
    def w2s(self):
        word2stroke = {}
        with open('../data/util_data/word2piece.txt') as f:
            lines = f.readlines()
            for line in (lines):
                word = line[0]
                stroke = line[2:-1].split(',')
                word2stroke[word]=stroke
            word2stroke['<UNK>']=['<UNK>']
        return word2stroke
    
    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, eos='+'):
    
    '''char对应wordpiece，现在已经准好了wordpiece2index'''
    
    char_vocab = Vocab_stroke()
    char_vocab.feed('<PAD>')  # <PAD> is at index 0 in char vocab
    char_vocab.feed('<UNK>')  # <UNK> is at index 1 in char vocab
    '''一个词用<>套住wordpiece'''
    char_vocab.feed('{')      # start is at index 2 in char vocab
    char_vocab.feed('}')      # end   is at index 3 in char vocab

    word_vocab = Vocab_token()
    word_vocab.feed('<PAD>')  # <PAD> is at index 0 in char vocab
    word_vocab.feed('<UNK>')  # <UNK> is at index 1 in word vocab

    actual_max_word_length = 0

    def _flatten(myList):
        output = []
        for sublist in myList:
            for ele in sublist:
                output.append(ele)
        return output 
    
    def _is_Chinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    
    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)
    
    '''遍历所有数据进行处理'''
    for fname in ('train', 'valid', 'test'):
        print('>>> reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in tqdm(f):
                ''''''
                line = line.strip()
                line = line.replace('}', '').replace('{', '').replace('|', '')
                if eos:
                    line = line.replace(eos, '')
                #顺序遍历当前句子的所有词
                for word in line.split():
                    
                    '''
                    # 英文对当前char长度进行限制  中文不需要
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length-2]
                    '''
                    
                    '''将所有出现的词加入词库'''
                    word_tokens[fname].append(word_vocab.feed(word))
                    
                    '''判断当前token是字还是词'''
                    if len(word) != 1:#词或<DIGIT>
                        if word == '<DIGIT>':
                            '''将<DIGIT>加入wordpiece 并返回 { <DIGIT> } 的index'''
                            char_array = [char_vocab.feed(c) for c in ['{','<DIGIT>','}']]
                            char_tokens[fname].append(char_array)
                        else:#其他词
                            strokes = []#wordpiece的list
                            '''拿到词中的每一个字符'''
                            for char in word:
                                if char in char_vocab.word_stroke_dict:#字符在词库里
                                    '''拿到当前char的wordpiece'''
                                    strokes.append(char_vocab.word_stroke_dict[char])
                                else:strokes.append(['<UNK>'])
                            # [[wordpiece1],[wordpiece2]] --> [wordpieces]        
                            temp = _flatten(strokes)
                            temp = ['{'] + temp
                            '''wordpiece2index'''
                            char_array = [char_vocab.feed(c) for c in temp]
                            '''{ + 最多31个wordpiece'''
                            char_array = char_array[:31]
                            char_array.append(char_vocab.feed('}'))
                            '''{ + 最多30个wordpiece + }  ---> 最长32字符'''
                            char_tokens[fname].append(char_array)
                    
                    else:#单个字的情况
                        '''判是不是中文'''
                        if _is_Chinese(word):# 中文
                            '''{ + 字 + }'''
                            # try:
                            strokes = ['{']+char_vocab.word_stroke_dict[word]+['}']
                            char_array = [char_vocab.feed(c) for c in strokes]
                            char_tokens[fname].append(char_array)
                            # except:
                            #     print(word)
                            #     strokes = ['{']+char_vocab.word_stroke_dict['<UNK>']+['}']
                            #     char_array = [char_vocab.feed(c) for c in strokes]
                            #     char_tokens[fname].append(char_array)
                        else: # 符号
                            char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                            char_tokens[fname].append(char_array)
        

                if eos:
                    '''将 结束 符号加入词库'''
                    word_tokens[fname].append(word_vocab.feed(eos))
                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    print()
    print('>>> size of word vocabulary:', word_vocab.size)
    print('>>> size of wordpiece vocabulary:', char_vocab.size)
    print('>>> number of tokens in train:', len(word_tokens['train']))
    print('>>> number of tokens in valid:', len(word_tokens['valid']))
    print('>>> number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), 32], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, 32


class DataReader:

    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps):

        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]

        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


if __name__ == '__main__':

    _, _, wt, ct, _ = load_data('data', 32)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid'], ct['valid'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
