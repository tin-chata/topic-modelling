"""
Created on 2019-01-07
@author: duytinvo
"""
import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
from core_nns import Autoencoder
from other_utils import Progbar, Timer, SaveloadHP
from data_utils import Vocab, Data2tensor, Txtfile, seqPAD, Embeddings, PADt

Data2tensor.set_randseed(1234)


class Autoencoder_model(object):
    def __init__(self, args=None):

        self.args = args
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")

        self.word2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.w2i, unk_words=True,
                                               se_words=self.args.start_end)

        word_HPs = [len(self.args.vocab.w2i), self.args.word_dim, self.args.word_pretrained, self.args.word_drop_rate,
                    self.args.word_zero_padding, self.args.grad_flag, self.args.word_nn_out_dim]

        self.model = Autoencoder(HPs=word_HPs).to(self.device)

        if args.optimizer.lower() == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)

    def train_batch(self, train_data):
        clip_rate = self.args.clip
        chunk_size = self.args.batch_size * (self.args.neg_samples + 1)
        total_batch = self.args.vocab.nodocs // chunk_size
        prog = Progbar(target=total_batch)

        # set model in train model
        train_loss = []
        self.model.train()

        for i, inp_ids in enumerate(self.args.vocab.minibatches(train_data, batch_size=chunk_size)):
            padded_inp, _ = seqPAD.pad_sequences(inp_ids, pad_tok=self.args.vocab.w2i[PADt])
            data_tensor = Data2tensor.idx2tensor(padded_inp, torch.long, self.device)

            # shuffle data_chunks
            perm_ids = torch.randperm(chunk_size)
            data_tensor = data_tensor[perm_ids]
            data_tensor = data_tensor.view(self.args.batch_size, self.args.neg_samples + 1, -1)
            # data_tensor = [batch_size, 1 + neg_sampling, word_length]
            inp_tensor = data_tensor[:, 0, :]
            noise_tensor = data_tensor[:, 1:, :]

            self.model.zero_grad()
            emb_sent, trans_sent, emb_noise = self.model(inp_tensor, noise_tensor)

            batch_loss = self.model.batchHingeLoss(emb_sent, trans_sent, emb_noise)
            train_loss.append(batch_loss.item())

            batch_loss.backward()
            if clip_rate > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_rate)
            self.optimizer.step()
            prog.update(i + 1, [("Train loss", batch_loss.item())])

        return np.mean(train_loss)

    def lr_decay(self, epoch):
        lr = self.args.lr / (1 + self.args.decay_rate * epoch)
        print("INFO: - Learning rate is setted as: %f" % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        train_data = Txtfile(self.args.train_file, firstline=False, word2idx=self.word2idx, limit=self.args.sent_limit)
        model_filename = os.path.join(args.model_dir, self.args.model_file)
        max_epochs = self.args.max_epochs
        epoch_start = time.time()
        for epoch in range(1, max_epochs + 1):
            print("Epoch: %s/%s" % (epoch, max_epochs))
            train_loss = self.train_batch(train_data)

            print("UPDATES: - Train loss: %.4f" % train_loss)
            print("         - Save the model to %s at epoch %d" % (model_filename, epoch))
            # Convert model to CPU to avoid out of GPU memory
            self.model.to("cpu")
            torch.save(self.model.state_dict(), model_filename)
            self.model.to(self.device)

            epoch_finish = Timer.timeEst(epoch_start, epoch / max_epochs)
            print("\nINFO: - Trained time (Remained time for %d epochs): %s" % (max_epochs - epoch, epoch_finish))

            if self.args.decay_rate > 0:
                self.lr_decay(epoch)

        word_emb, enc_emb, dec_emb = self.model.get_embs()
        id2topic = {}
        for i in range(enc_emb.shape[0]):
            id2topic[i] = "topic_%d" % i

        Embeddings.save_embs(id2topic, dec_emb.transpose(), os.path.join(args.model_dir,self.args.dtopic_emb_file))
        Embeddings.save_embs(id2topic, enc_emb, os.path.join(args.model_dir,self.args.etopic_emb_file))
        Embeddings.save_embs(self.args.vocab.i2w, word_emb, os.path.join(args.model_dir,self.args.tuned_word_emb_file))
        return

    @staticmethod
    def build_data(args):
        print("Building dataset...")
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        vocab = Vocab(wl_th=args.wl_th, wcutoff=args.wcutoff)

        idf_file = os.path.join(args.model_dir, args.idf_file)
        vocab.build(fname=args.train_file, idf_file=idf_file, firstline=False, limit=args.sent_limit)
        args.vocab = vocab
        if len(args.word_emb_file) != 0:
            scale = np.sqrt(3.0 / args.word_dim)
            args.word_pretrained = Embeddings.get_W(args.word_emb_file, args.word_dim, vocab.w2i, scale)
        else:
            args.word_pretrained = None

        SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))
        return args


if __name__ == '__main__':
    """
    """
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--train_file', help='Trained file', type=str,
                           default="/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.pro.txt")

    argparser.add_argument("--tuned_word_emb_file", type=str, help="Word embedding file after fine-tuning",
                           default="word_emb.txt")

    argparser.add_argument("--dtopic_emb_file", type=str, help="Topic embedding file after fine-tuning",
                           default="dtopic_emb.txt")

    argparser.add_argument("--etopic_emb_file", type=str, help="Inverse topic embedding file after fine-tuning",
                           default="etopic_emb.txt")

    argparser.add_argument("--wl_th", type=int, default=None, help="Word threshold")

    argparser.add_argument("--wcutoff", type=int, default=5, help="Prune words occurring <= wcutoff")

    argparser.add_argument("--start_end", action='store_true', default=False, help="Start-end padding flag")

    argparser.add_argument("--idf_file", type=str, help="tfidf file", default="idf.txt")

    argparser.add_argument("--word_emb_file", type=str, help="Word embedding file", default="")

    argparser.add_argument("--word_dim", type=int, default=100, help="Word embedding size")

    argparser.add_argument("--word_drop_rate", type=float, default=0.5,
                           help="Dropout rate at word-level embedding")

    argparser.add_argument("--word_zero_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at word level")

    argparser.add_argument("--grad_flag", action='store_false', default=True, help="Gradient emb flag (default True)")

    argparser.add_argument("--word_nn_out_dim", type=int, default=15, help="Word-level neural network dimension")

    argparser.add_argument("--optimizer", type=str, default="ADAM", help="Optimized method (adagrad, sgd, ...)")

    argparser.add_argument("--lr", type=float, default=0.002, help="Learning rate")

    argparser.add_argument("--decay_rate", type=float, default=0.05, help="Decay rate")

    argparser.add_argument("--max_epochs", type=int, default=64, help="Maximum trained epochs")

    argparser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")

    argparser.add_argument("--neg_samples", type=int, default=8, help="Number of negative samples")

    argparser.add_argument('--clip', default=5, type=int, help='Clipping value')

    argparser.add_argument('--model_dir', help='Model directory', default="./extracted_data/", type=str)

    argparser.add_argument('--model_file', help='Trained model filename', default="tpmd.m", type=str)

    argparser.add_argument('--model_args', help='Trained argument filename', default="tpmd.args", type=str)

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument("--sent_limit", type=int, default=10000, help="Limit the number of lines to train")

    args = argparser.parse_args()

    args = Autoencoder_model.build_data(args)

    topic_encoder = Autoencoder_model(args)

    topic_encoder.train()

    # os.system("sudo shutdown +1")
