"""
Created on 2019-01-15
@author: duytinvo
"""
import os
import torch
from model import Autoencoder_model
from other_utils import SaveloadHP
from data_utils import Txtfile, seqPAD, PADt, Data2tensor

use_cuda=False
model_args = "extracted_data/tpmd.args"

margs = SaveloadHP.load(model_args)
margs.use_cuda = use_cuda

model_filename = os.path.join(margs.model_dir, margs.model_file)
print("Load Model from file: %s" % model_filename)
topic_encoder = Autoencoder_model(margs)
topic_encoder.model.load_state_dict(torch.load(model_filename))
topic_encoder.model.to(topic_encoder.device)


def inference(model, rv, auxiliary_embs=None):
    pro_rv = Txtfile.process_sent(rv)
    rv_id = topic_encoder.word2idx(pro_rv)
    padded_inp, _ = seqPAD.pad_sequences([rv_id], pad_tok=margs.vocab.w2i[PADt])
    inputs = Data2tensor.idx2tensor(padded_inp, torch.long, topic_encoder.device)

    with torch.no_grad():
        model.eval()
        # inputs = [batch_size, sent_length]
        # auxiliary_embs = [batch_size, sent_length, aux_dim]
        emb_word = model.emb_layer(inputs, auxiliary_embs)
        # emb_word = [batch_size, sent_length, emb_dim]
        emb_sent = emb_word.mean(dim=1, keepdim=True)
        # emb_sent = [batch_size, 1, emb_dim]
        sent_length = emb_word.size(1)
        emb_sent_ex = emb_sent.expand(-1, sent_length, -1).contiguous()
        # emb_sent_ex = [batch_size, sent_length, emb_dim]
        alpha_score = model.attention(emb_word, emb_sent_ex)
        # alpha_score = [batch_size, sent_length, 1]
        alpha_norm = model.norm_attention(alpha_score)
        # alpha_norm = [batch_size, sent_length, 1]
        emb_attsent = torch.bmm(alpha_norm.transpose(1, 2), emb_word)
        # emb_attsent = [batch_size, 1, emb_dim] <------
        # alpha_norm.transpose(1, 2) = [batch_size, 1, sent_length] dot emb_word = [batch_size, sent_length, emb_dim]
        emb_topic = model.encoder(emb_attsent.squeeze(1))
        topic_class = model.norm_layer(emb_topic)
        # emb_topic = topic_class = [batch_size, nn_out_dim]
        label_prob, label_pred = topic_class.data.topk(topic_class.size(1))
        return label_prob, label_pred


if __name__ == '__main__':

    rv = "back in 2005 2007 this place was my favorite thai place ever id go here alllll the time i never had any " \
         "complaints once they started to get more known and got busy their service started to suck and their portion " \
         "sizes got cut in half i have a huge problem with paying more for way less food the last time i went there i "

    # label_prob, label_pred = inference(topic_encoder.model, rv, auxiliary_embs=None)

    label_prob, label_pred = topic_encoder.predict(rv)


