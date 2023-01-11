
import torch
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
import pickle

def dump_model_data(data_dict, path):
    """dump model details to be loaded during eval and inference
    """
    with open(path, 'wb') as prefile:
        pickle.dump(data_dict, prefile)

def set_seed():
    """set random seed to reproduce results
    """
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def load_data(path):
    """load train/test csv files
    """
    df = pd.read_csv(path, sep='\t', names=['text', 'intent'])
    return df

def add_multilabel_columns(df, column_names):
    """ add columns to dataframe for multilabel classification
    """
    for col in column_names:
        df[col] = df.apply(lambda x: 1 if col in x['intent'].split('+') else 0, axis =1)
    return df

def get_unique_labels(df):
    """ extract all the unique labels from dataframe
    """
    intent_split = df.apply(lambda x: tuple(x['intent'].split('+')), axis =1)
    labels_list = intent_split.unique()
    labels_list = set([label for label_comb in labels_list for label in label_comb])
    return list(labels_list)

def split_dataset(df, labels_list):
    """split data into train, val using stratification for multilabel
    """
    df_index = np.expand_dims(df.index.to_numpy(), axis=1)
    y = np.expand_dims(df[labels_list].index.to_numpy(), axis=1)
    X_train, y_train, X_val, y_val = iterative_train_test_split(df_index, y, test_size = 0.2)
    X_train = df.loc[X_train[:,0]]
    X_val = df.loc[X_val[:,0]]
    return X_train, X_val

def preprocess_data(train_df, test_df):
    """preprocess for training, extract labels, split dataset,
    add multilabel columns,
    """
    labels_list = get_unique_labels(train_df)
    train_df = add_multilabel_columns(train_df, labels_list)
    train_df, val_df = split_dataset(train_df, labels_list)
    test_df = add_multilabel_columns(test_df, labels_list)
    return train_df, val_df, test_df, labels_list

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    
def get_class_weights(X_train, labels_list):
    '''
    Create weights for class based on data distribution.
    To be used in weighted cross entropy.
    '''
    N = len(X_train)
    class_weights = {}
    positive_weights = {}
    negative_weights = {}
    for ind, label in enumerate(labels_list):
        positive_weights[ind] = clamp((N/10) /(2 * sum(X_train[label] == 1)), 0.1, 3.0)
        negative_weights[ind] = clamp((N/10) /(2 * sum(X_train[label] == 0)), 0.1, 3.0)

    class_weights['positive_weights'] = positive_weights
    class_weights['negative_weights'] = negative_weights
    print(class_weights)
    print(labels_list)
    return class_weights



