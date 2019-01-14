#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:31:21 2018

@author: dtvo
"""
import re
import os
import random
import torch
import itertools
import numpy as np
from collections import Counter

# ----------------------
#    Word symbols
# ----------------------
PADt = u"<PADt>"
SOt = u"<st>"
EOt = u"</st>"
UNKt = u"<UNKt>"


# ----------------------------------------------------------------------------------------------------------------------
# ======================================== DATA-RELATED FUNCTIONS ======================================================
# ----------------------------------------------------------------------------------------------------------------------
class Vocab(object):
    def __init__(self, wl_th=None, wcutoff=1):
        self.idf = Counter()
        self.wcnt = Counter()
        self.i2w = {}
        self.w2i = {}
        self.wl = wl_th
        self.wcutoff = wcutoff
        self.nodocs = 0

    def build(self, fname, idf_file, firstline=False, limit=-1):
        """
        Read a list of file names, return vocabulary
        :param files: list of file names
        :param firstline: ignore first line flag
        :param limit: read number of lines
        """
        wl = 0
        count = 0

        raw = Txtfile(fname, firstline=firstline, limit=limit)
        for sent in raw:
            count += 1
            self.wcnt.update(sent)
            self.idf.update(set(sent))
            wl = max(wl, len(sent))

        wlst = [PADt, SOt, EOt, UNKt] + [x for x, y in self.wcnt.most_common() if y >= self.wcutoff]
        self.w2i = dict([(y, x) for x, y in enumerate(wlst)])
        self.i2w = dict([(x, y) for x, y in enumerate(wlst)])
        self.wl = wl if self.wl is None else min(wl, self.wl)
        self.nodocs = count

        if os.path.exists(idf_file):
            self.loadidf(idf_file)
        else:
            for wd in self.idf:
                self.idf[wd] = np.log(self.nodocs / self.idf[wd])
            self.writeidf(idf_file)

        print("Extracting vocabulary: %d total input samples" % count)
        print("\t%d total words" % sum(self.wcnt.values()))
        print("\t%d unique words" % len(self.wcnt))
        print("\t%d unique words appearing at least %d times" % (len(self.w2i)-4, self.wcutoff))

    def writeidf(self, fname):
        with open(fname, "w") as f:
            f.write("%d %d\n" % (len(self.idf), 1))
            for wd in self.idf:
                f.write("%s %f\n" % (wd, self.idf[wd]))

    def loadidf(self, fname):
        with open(fname, "r") as f:
            for line in f:
                wd, v = line.strip().split()
                self.idf[wd] = float(v)

    @staticmethod
    def wd2idx(vocab_words=None, unk_words=True, se_words=False):
        """
        Return a function to convert tag2idx or word/word2idx
        """
        def f(sent):
            if vocab_words is not None:
                # SOw,EOw words for  SOW
                word_ids = []
                for word in sent:
                    # ignore words out of vocabulary
                    if word in vocab_words:
                        word_ids += [vocab_words[word]]
                    else:
                        if unk_words:
                            word_ids += [vocab_words[UNKt]]
                        else:
                            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if se_words:
                    # SOc,EOc words for  EOW
                    word_ids = [vocab_words[SOt]] + word_ids + [vocab_words[EOt]]
                    # 4. return tuple char ids, word id
            return word_ids
        return f

    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)

        Yields:
            list of tuples
        """
        x_batch = []
        for x in data:
            if len(x_batch) == batch_size:
                yield x_batch
                x_batch = []

            if type(x[0]) == tuple:
                x = list(zip(*x))
            x_batch += [x]

        # if len(x_batch) != 0:
        #     yield x_batch


class Txtfile(object):
    """
    Read cvs file
    """
    def __init__(self, fname, word2idx=None, firstline=True, limit=-1):
        self.fname = fname
        self.firstline = firstline
        self.limit = limit if limit > 0 else None
        self.word2idx = word2idx
        self.length = None
        
    def __iter__(self):
        with open(self.fname, newline='', encoding='utf-8') as f:
            f.seek(0)
            if self.firstline:
                # Skip the header
                next(f)
            for line in itertools.islice(f, self.limit):
                sent = Txtfile.process_sent(line)
                if self.word2idx is not None:
                    sent = self.word2idx(sent)
                yield sent
                
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def process_sent(sent):
        sent = sent.strip().split()
        # TODO: add a text-preprocessor at both word-level and character-level to improve performance
        sent = re.sub('[^a-zA-Z ]+', '', sent)
        sent = sent.lower()
        return sent


class seqPAD:
    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
    
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []
    
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]
    
        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=-1, cthres=-1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
            nlevels: "depth" of padding, for the case where we have word ids
    
        Returns:
            a list of list where each sublist has same length
    
        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            max_length = min(wthres, max_length) if wthres > 0 else max_length
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)
    
        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min(cthres, max_length_word) if cthres > 0 else max_length_word
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = min(wthres, max_length_sentence) if wthres > 0 else max_length_sentence
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
            # set sequence length to 1 by inserting padding 
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)
    
        return sequence_padded, sequence_length


class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, dtype=torch.long, device=torch.device("cpu")):
        vec = torch.tensor(indexes, dtype=dtype, device=device)
        return vec

    @staticmethod
    def sort_tensors(label_ids, word_ids, sequence_lengths, char_ids=None, word_lengths=None,
                     dtype=torch.long, device=torch.device("cpu")):
        label_tensor = Data2tensor.idx2tensor(label_ids, dtype, device)
        word_tensor = Data2tensor.idx2tensor(word_ids, dtype, device)
        sequence_lengths = Data2tensor.idx2tensor(sequence_lengths, dtype, device)
        sequence_lengths, word_perm_idx = sequence_lengths.sort(0, descending=True)
        word_tensor = word_tensor[word_perm_idx]
        label_tensor = label_tensor[word_perm_idx]
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)

        if char_ids is not None:
            char_tensor = Data2tensor.idx2tensor(char_ids, dtype, device)
            word_lengths = Data2tensor.idx2tensor(word_lengths, dtype, device)
            batch_size = len(word_ids)
            max_seq_len = sequence_lengths.max()
            char_tensor = char_tensor[word_perm_idx].view(batch_size * max_seq_len.item(), -1)
            word_lengths = word_lengths[word_perm_idx].view(batch_size * max_seq_len.item(), )
            word_lengths, char_perm_idx = word_lengths.sort(0, descending=True)
            char_tensor = char_tensor[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        else:
            char_tensor = None
            word_lengths = None
            char_seq_recover = None
        return label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover

    @staticmethod
    def set_randseed(seed_num=12345):
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)


class Embeddings:
    @staticmethod
    def save_embs(i2w, data, fname):
        with open(fname, 'w') as f:
            V, s = data.shape
            f.write("%d %d\n" % (V, s))
            for i in range(V):
                emb = " ".join([str(j) for j in data[i]])
                f.write("%s %s\n" % (i2w[i], emb))

    @staticmethod
    def load_embs(fname):
        embs = []
        wd2id = {}
        with open(fname, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    continue
                else:
                    w = "".join(p[0])
                    wd2id[w] = len(wd2id)
                    e = [float(i) for i in p[1:]]
                    embs.append(e)
        return wd2id, np.array(embs, dtype="float32")

    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pre-trained embeddings:")
        wd2id, word_vecs = Embeddings.load_embs(emb_file)
        print('\t%d pre-trained embeddings' % (len(wd2id)))
        print('Mapping to vocabulary:')
        unk = 0
        part = 0
        full = 0
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.items():
            if word in [PADt, SOt, EOt, UNKt]:
                continue
            if wd2id.get(word, -1) != -1:
                W[idx] = word_vecs[wd2id[word]]
                full += 1
            else:
                if wd2id.get(word.lower(), -1) != -1:
                    W[idx] = word_vecs[wd2id[word.lower()]]
                    part += 1
                else:
                    unk += 1
                    rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
                    W[idx] = rvector
        print('\t%d non-matching (random) vectors;' % unk)
        print('\t%d partially matching (case-insensitive) vectors;' % part)
        print('\t%d fully-matching (case-sensitive) word vectors.' % full)
        return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.iteritems():
            if idx == 0:
                continue
            W[idx] = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
        return W


if __name__ == "__main__":
    filename = "/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.pro.txt"
    idf_file = "./extracted_data/idf.txt"
    vocab = Vocab(wl_th=None, wcutoff=5)
    vocab.build(filename, idf_file, firstline=False, limit=100000)

    word2idx = vocab.wd2idx(vocab_words=vocab.w2i, unk_words=True, se_words=False)

    train_data = Txtfile(filename, firstline=False, word2idx=word2idx, limit=100000)

    batch_size = 8
    neg_sampling = 5
    no_chunks = batch_size * (neg_sampling + 1)
    train_iters = Vocab.minibatches(train_data, batch_size=no_chunks)

    for inp_ids in train_iters:
        padded_inp, _ = seqPAD.pad_sequences(inp_ids, pad_tok=vocab.w2i[PADt])
        data_tensor = Data2tensor.idx2tensor(padded_inp)

        # shuffle chunks
        perm_ids = torch.randperm(no_chunks)
        data_tensor = data_tensor[perm_ids]
        data_tensor = data_tensor.view(batch_size, neg_sampling + 1, -1)

        inp_tensor = data_tensor[:, 0, :]
        noise_tensor = data_tensor[:, 1:, :]
        break

    scale = np.sqrt(3.0 / 100)
    trained_vectors = Embeddings.get_W("extracted_data/extracted_data/w2v_yelp100.pro.vec", 100, vocab.w2i, scale)
