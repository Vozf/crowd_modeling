import numpy as np
from gensim.models import KeyedVectors

# download link for GoogleNews-vectors-negative300.bin: https://www.kaggle.com/datasets/sandreds/googlenewsvectorsnegative300?resource=download

def main():
    cutoff = 10000
    news_path = './GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
    embeddings = embeddings_index.vectors
    print(embeddings.shape)
    np.save("./news.npy", embeddings)
    np.save("./news_subset.npy", embeddings[:cutoff])


if __name__ == '__main__':
    main()
