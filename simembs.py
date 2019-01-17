import argparse
import numpy as np

# ----------------------
#    Word symbols
# ----------------------
PADt = u"<PADt>"
SOt = u"<st>"
EOt = u"</st>"
UNKt = u"<UNKt>"


def load_idf(fname):
    idf = dict()
    with open(fname, 'r') as f:
        next(f)
        for line in f:
            p = line.strip().split()
            # assert len(p)== s+1
            w = "".join(p[0])
            idf[w] = float(p[1])
    return idf


def load_embs(emb_file, idf=dict()):
    with open(emb_file, 'r') as f:
        vectors = []
        id2wd = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) == 2:
                continue
            wd = vals[0]
            val = [float(x)*idf.get(wd, 1.0) for x in vals[1:]]
            if wd in [PADt, SOt, EOt, UNKt] or sum(val) == 0:
                continue
            id2wd[len(id2wd)] = wd
            vectors.append(val)
    return id2wd, np.array(vectors, dtype="float32")


def cosine_distance(W, ivocab, topic_file, N=100):
    # normalize each word vector to unit variance
    d = (np.sum(W ** 2, 1) ** 0.5)
    # vec(a) <--- vec(a)/||vec(a)||
    W_norm = (W.T / d).T

    id2topic, topic_embs = load_embs(topic_file)
    for i in range(len(id2topic)):
        vec_result = topic_embs[i]
        d = (np.sum(vec_result ** 2,) ** 0.5)
        # vec(b) <--- vec(b)/||vec(b)||
        vec_norm = (vec_result.T / d).T
        dist = np.dot(W_norm, vec_norm.T)
        # dist = np.dot(W, vec_result.T)
        a = np.argsort(-dist)[:N]
        neighbors = " ".join(["%s (%.4f);" % (ivocab[x], dist[x]) for x in a])
        print("%s: %s\n" % (id2topic[i], neighbors))


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--idf_file', default='extracted_data/idf.txt', type=str)
    parser.add_argument('--emb_file', default='extracted_data/word_emb.txt', type=str)
    parser.add_argument('--topic_file', default='extracted_data/dtopic_emb.txt', type=str)
    parser.add_argument('--N', default=50, type=int)
    args = parser.parse_args()
    idf = load_idf(args.idf_file)
    ivocab, W = load_embs(args.emb_file, idf)
    cosine_distance(W, ivocab, args.topic_file, args.N)

