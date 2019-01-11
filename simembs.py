import argparse
import numpy as np


def generate(emb_file):
    with open(emb_file, 'r') as f:
        vectors = {}
        words = []
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) == 2:
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            words.append(vals[0])
            
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term, N=100):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        if term in vocab:
            index = vocab[term]
            dist[index] = -np.Inf
        else:
            continue

    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


def load_embs(fname):
    embs = dict()
    s = 0
    V = 0
    with open(fname, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2:
                V = int(p[0])  # Vocabulary
                s = int(p[1])  # embeddings size
            else:
                # assert len(p)== s+1
                w = "".join(p[0])
                # print(p)
                e = [float(i) for i in p[1:]]
                embs[w] = np.array(e, dtype="float32")
    #        assert len(embs) == V
    return embs


def topicdistance(W, vocab, ivocab, topic_file, N=100):
    topic_embs = load_embs(topic_file)
    for topic in topic_embs:
        vec_result = topic_embs[topic]
        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2,) ** (0.5))
        vec_norm = (vec_result.T / d).T

        dist = np.dot(W, vec_norm.T)
        a = np.argsort(-dist)[:N]

        neighbors = " ".join(["%s (%.4f);" % (ivocab[x], dist[x]) for x in a])
        print("%s: %s\n" % (topic, neighbors))


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', default='data/data/glove_yelp100.pro.vec.txt', type=str)
    parser.add_argument('--topic_file', default='data/data/dtopic_emb.txt', type=str)
    parser.add_argument('--N', default=40, type=int)
    args = parser.parse_args()
    W, vocab, ivocab = generate(args.emb_file)
    topicdistance(W, vocab, ivocab, args.topic_file, args.N)

