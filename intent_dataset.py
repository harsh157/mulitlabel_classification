import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    X, y = [], []
    seq_lens = []
    features = []
    for train in batch:
        if train['encoding'] is not None:
            seq_lens.append(len(train['encoding']))
            X.append(train['encoding'])
        else:
            features.append(train['feature'])
        y.append(train['labels'])
    
    seq_lens = torch.from_numpy(np.array(seq_lens, dtype=np.int64))
    y = torch.from_numpy(np.array(y, dtype=np.int64))
    padded_input = []
    if len(X) > 0:
        padded_input = pad_sequence(X, batch_first=True)

    features = torch.from_numpy(np.array(features, dtype=np.float)).to(torch.float32)

    return padded_input, y, seq_lens, features

class IntentDataset(Dataset):
    
    def __init__(self, data, label_columns, tokenizer, max_token_len=50, vectorizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data.sort_values(by='text', key=lambda col: [len(colval.split(' ')) for colval in col], ascending=False, ignore_index=True)
        
        #self.vocab = train_vocab
        self.label_columns = label_columns
        self.vectorizer = vectorizer
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[self.label_columns]
        
        #text = '<s> {} </s>'.format(self.data[0][index])
        #labels = self.data[1][index]
        
        if self.vectorizer:
            encoding = None
            feature = self.vectorizer.vectorize(text)
        else:
            encoding = torch.tensor(self.tokenizer.text_to_tokenids(text), dtype=torch.int64)
            feature = None
        return dict(text=text, encoding = encoding, labels = labels, feature=feature)
    
    def __len__(self):
        return len(self.data)


