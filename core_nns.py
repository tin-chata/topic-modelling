# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:41:43 2018
@author: dtvo
"""
import torch
import numpy as np
import torch.nn as nn


class Embs(nn.Module):
    """
    This module take (characters or words) indices as inputs and outputs (characters or words) embedding
    """
    def __init__(self, HPs):
        super(Embs, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding, grad_flag] = HPs
        self.zero_padding = zero_padding
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))

        self.embeddings.weight.requires_grad = grad_flag  # fix word embeddings
        self.drop = nn.Dropout(drop_rate)

    def get_embs(self, inputs, auxiliary_embs=None):
        """
        embs.shape([0, 1]) == auxiliary_embs.shape([0, 1])
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0, 1, 2, 3])
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        if auxiliary_embs is not None:
            assert embs_drop.shape[:-1] == auxiliary_embs.shape[:-1]
            embs_drop = torch.cat([embs_drop, auxiliary_embs], -1)
        return embs_drop

    def forward(self, inputs, auxiliary_embs=None):
        return self.get_embs(inputs, auxiliary_embs)

    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index, :] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def set_zeros(self, idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)


class Autoencoder(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """

    def __init__(self, HPs):
        super(Autoencoder, self).__init__()
        [emb_size, emb_dim, pre_embs, emb_drop_rate, emb_zero_padding, grad_flag, nn_out_dim] = HPs

        emb_HPs = [emb_size, emb_dim, pre_embs, emb_drop_rate, emb_zero_padding, grad_flag]
        self.emb_layer = Embs(emb_HPs)
        self.attention = nn.Bilinear(emb_dim, emb_dim, 1)
        self.norm_attention = nn.Softmax(1)

        self.encoder = nn.Linear(emb_dim, nn_out_dim)
        self.norm_layer = nn.Softmax(-1)
        self.decoder = nn.Linear(nn_out_dim, emb_dim)

    def forward(self, inputs, noises, aux_inputs=None, aux_noises=None):
        emb_sent, trans_sent = self.get_attinput_features(inputs, aux_inputs)
        emb_noise = self.get_noise_features(noises, aux_noises)
        return emb_sent, trans_sent, emb_noise

    def get_input_features(self, inputs, auxiliary_embs=None):
        # inputs = [batch_size, sent_length]
        # auxiliary_embs = [batch_size, sent_length, aux_dim]
        emb_word = self.emb_layer(inputs, auxiliary_embs)
        # emb_word = [batch_size, sent_length, emb_dim]
        emb_sent = emb_word.mean(dim=1)
        # emb_sent = [batch_size, emb_dim]
        emb_topic = self.encoder(emb_sent)
        topic_class = self.norm_layer(emb_topic)
        # emb_topic = topic_class = [batch_size, nn_out_dim]
        trans_sent = self.decoder(topic_class)
        # trans_sent = [batch_size, emb_dim]
        return emb_sent, trans_sent

    def get_attinput_features(self, inputs, auxiliary_embs=None):
        # inputs = [batch_size, sent_length]
        # auxiliary_embs = [batch_size, sent_length, aux_dim]
        emb_word = self.emb_layer(inputs, auxiliary_embs)
        # emb_word = [batch_size, sent_length, emb_dim]
        emb_sent = emb_word.mean(dim=1, keepdim=True)
        # emb_sent = [batch_size, 1, emb_dim]
        sent_length = emb_word.size(1)
        emb_sent_ex = emb_sent.expand(-1, sent_length, -1).contiguous()
        # emb_sent_ex = [batch_size, sent_length, emb_dim]
        alpha_score = self.attention(emb_word, emb_sent_ex)
        # alpha_score = [batch_size, sent_length, 1]
        alpha_norm = self.norm_attention(alpha_score)
        # alpha_norm = [batch_size, sent_length, 1]
        emb_attsent = torch.bmm(alpha_norm.transpose(1, 2), emb_word)
        # emb_attsent = [batch_size, 1, emb_dim] <------
        # alpha_norm.transpose(1, 2) = [batch_size, 1, sent_length] dot emb_word = [batch_size, sent_length, emb_dim]
        emb_topic = self.encoder(emb_attsent.squeeze(1))
        topic_class = self.norm_layer(emb_topic)
        # emb_topic = topic_class = [batch_size, nn_out_dim]
        trans_sent = self.decoder(topic_class)
        # trans_sent = [batch_size, emb_dim]
        return emb_attsent.squeeze(1), trans_sent

    def inference(self, inputs, auxiliary_embs=None):
        # inputs = [batch_size, sent_length]
        # auxiliary_embs = [batch_size, sent_length, aux_dim]
        emb_word = self.emb_layer(inputs, auxiliary_embs)
        # emb_word = [batch_size, sent_length, emb_dim]
        emb_sent = emb_word.mean(dim=1, keepdim=True)
        # emb_sent = [batch_size, 1, emb_dim]
        sent_length = emb_word.size(1)
        emb_sent_ex = emb_sent.expand(-1, sent_length, -1).contiguous()
        # emb_sent_ex = [batch_size, sent_length, emb_dim]
        alpha_score = self.attention(emb_word, emb_sent_ex)
        # alpha_score = [batch_size, sent_length, 1]
        alpha_norm = self.norm_attention(alpha_score)
        # alpha_norm = [batch_size, sent_length, 1]
        emb_attsent = torch.bmm(alpha_norm.transpose(1, 2), emb_word)
        # emb_attsent = [batch_size, 1, emb_dim] <------
        # alpha_norm.transpose(1, 2) = [batch_size, 1, sent_length] dot emb_word = [batch_size, sent_length, emb_dim]
        emb_topic = self.encoder(emb_attsent.squeeze(1))
        topic_class = self.norm_layer(emb_topic)
        # emb_topic = topic_class = [batch_size, nn_out_dim]
        label_prob, label_pred = emb_topic.data.topk(emb_topic.size(1))
        return label_prob, label_pred

    def get_noise_features(self, noises, auxiliary_embs=None):
        # noises = [batch_size, sampling, sent_length]
        # auxiliary_embs = [batch_size, sampling, sent_length, aux_dim]
        emb_word = self.emb_layer(noises, auxiliary_embs)
        # emb_word = [batch_size, sampling, sent_length, emb_dim]
        emb_noise = emb_word.mean(dim=2)
        # emb_noise = [batch_size, sampling, emb_dim]
        return emb_noise

    def get_embs(self):
        word_emb = self.emb_layer.embeddings.weight.data.cpu().numpy()
        enc_emb = self.encoder.weight.data.cpu().numpy()
        dec_emb = self.decoder.weight.data.cpu().numpy()
        return word_emb, enc_emb, dec_emb

    def regularized(self):
        dec_emb = self.decoder.weight
        # dec_emb = [emb_dim, nn_out_dim]
        norm_emb = nn.functional.normalize(dec_emb, p=2, dim=0)
        reg = norm_emb.t().mm(norm_emb)
        id_mt = torch.eye(*reg.size(), out=torch.empty_like(reg))
        reg -= id_mt
        return reg.norm()

    def batchHingeLoss(self, emb_sent, trans_sent, emb_noise):
        # emb_sent = [batch_size, emb_dim]
        # trans_sent = [batch_size, emb_dim]
        # emb_noise = [batch_size, sampling, emb_dim]
        y_score = (emb_sent*trans_sent).sum(-1)
        # y_score = [batch_size]
        batch_size, emb_dim = trans_sent.size()
        pred_score = torch.bmm(emb_noise, trans_sent.view(batch_size, emb_dim, 1)).squeeze(-1)
        # pred_score = [batch_size, sampling]
        distance = 1 + pred_score - y_score.view(-1, 1)
        abs_distance = torch.max(distance, torch.zeros_like(distance))
        ranking = abs_distance.sum(-1)
        reg = self.regularized()
        return ranking.mean() + reg


if __name__ == "__main__":
    import random
    from data_utils import Data2tensor, Vocab, seqPAD, Txtfile, PADt
    Data2tensor.set_randseed(1234)
    use_cuda = torch.cuda.is_available()
    filename = "/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.pro.txt"
    idf_file = "./extracted_data/idf.txt"

    vocab = Vocab(wl_th=None, wcutoff=5)
    vocab.build(filename, idf_file=idf_file, firstline=False, limit=100000)

    word2idx = vocab.wd2idx(vocab_words=vocab.w2i, unk_words=True, se_words=False)

    train_data = Txtfile(filename, firstline=False, word2idx=word2idx, limit=100000)

    batch_size = 8
    neg_sampling = 5
    no_chunks = batch_size * (neg_sampling + 1)
    train_iters = Vocab.minibatches(train_data, batch_size=no_chunks)
    data = []
    label = []
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

    emb_size = len(vocab.w2i)
    emb_dim = 50
    pre_embs = None
    emb_drop_rate = 0.5
    emb_zero_padding = False
    grad_flag = True
    nn_out_dim = 10
    HPs = [emb_size, emb_dim, pre_embs, emb_drop_rate, emb_zero_padding, grad_flag, nn_out_dim]

    topic_encoder = Autoencoder(HPs=HPs)
    emb_sent, trans_sent, emb_noise = topic_encoder(inp_tensor, noise_tensor)

    batch_loss = topic_encoder.batchHingeLoss(emb_sent, trans_sent, emb_noise)

    word_emb, enc_emb, dec_emb = topic_encoder.get_embs()
    id2topic = {}
    for i in range(enc_emb.shape[0]):
        id2topic[i] = "topic_%d" % i
