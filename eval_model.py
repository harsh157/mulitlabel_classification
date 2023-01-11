
import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from intent_dataset import IntentDataset
from lightning_modules import IntentDataModule, IntentLSTMModel, IntentNNModel
from tokenizer import SimpleTokenizer,GloveTokenizer,TfIdfExtractor
import utils
import pickle


def validation_metrics(test_df, trained_model, label_columns, tokenizer, max_token_len, vectorizer, args):
    THRESHOLD = 0.5
    
    test_dataset = IntentDataset(
      test_df,
      label_columns=label_columns,
      tokenizer=tokenizer,
      max_token_len=max_token_len,
      vectorizer=vectorizer
    )
    
    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        if args.model_type == 'lstm':
            _, prediction = trained_model(
              item["encoding"].unsqueeze(dim=0).to('cpu'),
              torch.from_numpy(np.array([len(item["encoding"])], dtype=np.int64))
            )
        else:
            _, prediction = trained_model(
              torch.from_numpy(item["feature"]).float()
            )

        predictions.append(prediction.flatten())
        labels.append(item["labels"].tolist())

    predictions = torch.stack(predictions).detach().cpu()
    y_pred = predictions.numpy()
    y_true = np.array(labels)
    y_pred = np.where(y_pred > THRESHOLD, 1, 0)
    report = classification_report(
      y_true,
      y_pred,
      target_names=label_columns,
      zero_division=0
    )
    print(report)
    fig, ax = plt.subplots(2, 9, figsize=(20,10))
    ax=ax.ravel()
    cm = multilabel_confusion_matrix(y_true, y_pred)
    #print(cm.shape)
    for i in range(17):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i])
        #print(ax[i])
        
        disp.plot(cmap=plt.cm.Blues, ax=ax[i], colorbar=False)
        ax[i].set_title(label_columns[i])
    plt.savefig('cm.jpg')
    plt.show()
    #labels = torch.stack(labels).detach().cpu()

def eval_model(args):

    test_df = utils.load_data(args.test_path)
    with open(args.model_data_file, 'rb') as modelfile:
        model_data = pickle.load(modelfile)

    test_df = utils.add_multilabel_columns(test_df, model_data['labels_list'])

    if args.model_type == 'lstm':
        trained_model = IntentLSTMModel.load_from_checkpoint(
          model_data['model_path']
        )

    else:
        trained_model = IntentNNModel.load_from_checkpoint(
          model_data['model_path']
        )
    trained_model.eval()
    trained_model.freeze()
    metrics = validation_metrics(test_df, trained_model, model_data['labels_list'], model_data['tokenizer'], model_data['max_token_count'], model_data['vectorizer'], args)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', type=str, default='lstm', help='lstm|nn')
    arg_parser.add_argument('--test_path', type=str, default='data/atis/test.tsv', help='Path for test csv')
    arg_parser.add_argument('--model_data_file', type=str, default='model_lstm.pkl', help='Details to reload model')

    args = arg_parser.parse_args()
    eval_model(args)


if __name__ == '__main__':
    main()
