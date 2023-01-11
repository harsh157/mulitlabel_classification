
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


def validation_metrics(val_df, trained_model, label_columns, tokenizer, max_token_len, vectorizer, args):
    """
    Function to calculate metrics on validation dataset
    like F1 score, macor and micro avg, confusion matrix
    """
    THRESHOLD = 0.5
    
    val_dataset = IntentDataset(
      val_df,
      label_columns=label_columns,
      tokenizer=tokenizer,
      max_token_len=max_token_len,
      vectorizer=vectorizer
    )
    
    predictions = []
    labels = []

    for item in tqdm(val_dataset):
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

def run_training(args):
    """
    method to run training for the model based on the arguments
    """
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MAX_TOKEN_COUNT = args.max_seq_length

    # Set seed to reproduce results
    utils.set_seed()

    # Load Data
    train_df = utils.load_data(args.train_path)
    test_df = utils.load_data(args.test_path)

    pretrained_embedding = None
    vectorizer = None
    tokenizer = None
    if args.model_type == 'lstm':
        # Create custom vocab or load glove embeddings
        if args.embed_path:
            tokenizer = GloveTokenizer()
            pretrained_embedding = tokenizer.init_from_path(args.embed_path)
            embed_dim = tokenizer.embed_dim
        else:
            tokenizer = SimpleTokenizer()
            tokenizer.create_vocab(train_df.text.values)
            embed_dim = args.embed_dim
    else:
        # Create tfidf features using training data
        vectorizer = TfIdfExtractor(train_df.text.values, max_features=args.nn_feature_size)

    #Preprocess data, add multilabel columns, extract labels list
    train_df, val_df, test_df, labels_list =  utils.preprocess_data(train_df, test_df)

    # Create class weights using data distribution
    class_weights = utils.get_class_weights(train_df, labels_list)

    # Create lightning data module for loading data
    data_module = IntentDataModule(
      train_df,
      val_df,
      test_df,
      labels_list,
      tokenizer,
      vectorizer=vectorizer,
      batch_size=BATCH_SIZE,
      max_token_len=MAX_TOKEN_COUNT
    )

    steps_per_epoch=len(train_df) // BATCH_SIZE

    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5

    warmup_steps, total_training_steps

    if args.model_type == 'lstm':
        model = IntentLSTMModel(
          len(labels_list), len(tokenizer.train_vocab),
          embed_dim, args.lstm_hidden_dim, 
          args.lstm_num_layers, tokenizer.train_vocab['<pad>'],
          labels_list,
          class_weights=class_weights,
          n_training_steps=total_training_steps,
          n_warmup_steps=warmup_steps,
          pretrained_embedding = pretrained_embedding
        )
    else:
        model = IntentNNModel(len(labels_list),
                args.nn_feature_size,
                [args.nn_fc1_dims, args.nn_fc2_dims],
                labels_list,
                class_weights=class_weights,
                n_training_steps=total_training_steps,
                n_warmup_steps=warmup_steps
        )

    checkpoint_callback = ModelCheckpoint(
      dirpath="checkpoints",
      filename="best-checkpoint",
      save_top_k=1,
      verbose=True,
      monitor="val_loss",
      mode="min"
    )
    logger = TensorBoardLogger("lightning_logs", name="intent-labeling")
    
    trainer = pl.Trainer(
      logger=logger,
      callbacks=[checkpoint_callback],#early_stopping_callback],
      max_epochs=N_EPOCHS,
      accelerator=args.accelerator
      #progress_bar_refresh_rate=30
    )
    
    trainer.fit(model, data_module)

    if args.model_type == 'lstm':
        trained_model = IntentLSTMModel.load_from_checkpoint(
          trainer.checkpoint_callback.best_model_path
        )

    else:
        trained_model = IntentNNModel.load_from_checkpoint(
          trainer.checkpoint_callback.best_model_path
        )
    trained_model.eval()
    trained_model.freeze()
    metrics = validation_metrics(val_df, trained_model, labels_list, tokenizer, MAX_TOKEN_COUNT, vectorizer, args)
    utils.dump_model_data({
        'model_type': args.model_type,
        'model_path':trainer.checkpoint_callback.best_model_path,
        'labels_list':labels_list,
        'tokenizer': tokenizer,
        'max_token_count': MAX_TOKEN_COUNT,
        'vectorizer': vectorizer,
        'validation_metrics': metrics
    }, path=args.model_data_file)


def main():
    """
    Method to parse arguments and run training.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', type=str, default='lstm', help='lstm|nn')
    arg_parser.add_argument('--nn_fc1_dims', type=int, default=2000, help='First layer size of nn')
    arg_parser.add_argument('--nn_fc2_dims', type=int, default=500, help='Second layer size of nn')
    arg_parser.add_argument('--nn_feature_size', type=int, default=1000, help='Tfidf feature size')
    arg_parser.add_argument('--embed_dim', type=int, default=128, help='Embed size for word tokens.')
    arg_parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='hidden dimension for lstm')
    arg_parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of layers in lstm')
    arg_parser.add_argument('--train_path', type=str, default='data/atis/train.tsv', help='Path for train csv')
    arg_parser.add_argument('--test_path', type=str, default='data/atis/test.tsv', help='Path for test csv')
    arg_parser.add_argument('--epochs', type=int, default=30, help='Epochs to train.')
    arg_parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training.')
    arg_parser.add_argument('--max_seq_length', type=int, default=50, help='Maximum length of text sequence')
    arg_parser.add_argument('--embed_path', type=str, default=None, help='Text embedding path')
    arg_parser.add_argument('--model_data_file', type=str, default='model_lstm.pkl', help='Details to reload model')
    arg_parser.add_argument('--accelerator', type=str, default='cpu', help='cpu|gpu')

    args = arg_parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
