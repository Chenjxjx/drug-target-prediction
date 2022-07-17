from gensim.models import Word2Vec
import pandas as pd
import numpy as np


def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class read_dataset(object):

    def __init__(self, dir, N_gram):
        self.df = pd.read_csv(dir)
        self.N_gram = N_gram

    def __iter__(self):
        for seq in self.df.sequence.values:
            yield seq_to_kmers(seq, self.N_gram)

if __name__ == "__main__":

    dataset = read_dataset("../Data_embedding/protinformation_csv/All/pall_n.csv",3)
    model = Word2Vec(vector_size=200, window=3, min_count=1, workers=6,sg=1)
    model.build_vocab(dataset)
    model.train(dataset,epochs=30,total_examples=model.corpus_count)
    model.save("./Model_word2vec/word2vec_pall.model")

