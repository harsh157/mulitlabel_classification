from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

"""
Custom tokenizer using training data.
First creates vocab the generates tokenids
"""
class SimpleTokenizer:

    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

    def create_vocab(self, text_list):
        counter = Counter()
        for txt in text_list:
          counter.update(self.tokenizer(txt))
        self.train_vocab = vocab(counter, min_freq=1,
                specials=['<pad>', '<s>', '</s>', '<unk>'])

    def text_to_tokenids(self, text):
        text = '<s> {} </s>'.format(text)
        return [self.train_vocab[token] if token in self.train_vocab else self.train_vocab['<unk>'] for token in self.tokenizer(text)]


"""
Tokenizer based on GloVe embeddings
First creates vocab the generates tokenids
"""
class GloveTokenizer(SimpleTokenizer):

    def __init__(self):
        super().__init__()

    def init_from_path(self, embed_path):
        vocab,embeddings = [],[]
        with open(embed_path,'rt', encoding='utf-8') as fi:
            full_content = fi.read().strip().split('\n')

        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)
        vocab_npa = np.array(vocab)
        embs_npa = np.array(embeddings)

        vocab_npa = np.insert(vocab_npa, 0, '<pad>')
        vocab_npa = np.insert(vocab_npa, 1, '<unk>')
        pad_emb_npa = np.zeros((1,embs_npa.shape[1]))
        unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)

        embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

        self.train_vocab = {wrd: ind for ind, wrd in enumerate(vocab_npa)}
        self.pad_idx = 0
        self.embed_dim = embs_npa.shape[1]

        return embs_npa

    def text_to_tokenids(self, text):
        return [self.train_vocab[token] if token in self.train_vocab else self.train_vocab['<unk>'] for token in self.tokenizer(text)]

"""
Create tfidf based feature using training data
"""
class TfIdfExtractor:

    def __init__(self, text_list, max_features=1000):
        self.tfv = TfidfVectorizer(min_df=3,  max_features=max_features, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = 'english')
        self.tfv.fit(text_list)

    def vectorize(self, text):
        return self.tfv.transform([text])[0].toarray()[0]




