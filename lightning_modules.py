import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from intent_dataset import IntentDataset, collate_batch
from lstm_model import LSTMModel
from nn_model import NNModel
from custom_loss import WeightedBCELoss
from torchmetrics import Accuracy, F1Score, AUROC

"""
Lightning module for LSTM model
"""
class IntentLSTMModel(pl.LightningModule):
    
  def __init__(self, n_classes: int, vocab_size, embed_dim, hidden_dim, num_layers, padding_idx, labels_list, class_weights=None, n_training_steps=None, n_warmup_steps=None, pretrained_embedding=None):
    super().__init__()
    self.lstm_model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers, n_classes, padding_idx, pretrained_embedding = pretrained_embedding)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.labels_list = labels_list
    self.save_hyperparameters()
    self.criterion = WeightedBCELoss()
    self.class_weights = class_weights

  def forward(self, input_ids, seq_lens, labels=None):
    """Forward pass through the model
    """
    output = self.lstm_model(input_ids, seq_lens)
    output = torch.sigmoid(output)
    loss = 0
    
    if labels is not None:
        loss = self.criterion(output, labels.float(), self.class_weights)
        #loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    """Perform one train step
    """
    input_ids = batch[0]
    labels = batch[1]
    seq_lens = batch[2]
    loss, outputs = self(input_ids, seq_lens, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    """Perform one validation step
    """
    input_ids = batch[0]
    labels = batch[1]
    seq_lens = batch[2]
    loss, outputs = self(input_ids, seq_lens, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    """Perform one test step
    """
    input_ids = batch[0]
    labels = batch[1]
    seq_lens = batch[2]
    loss, outputs = self(input_ids, seq_lens, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    """Dump AUROC on epoch end
    """
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()

    predictions = torch.stack(predictions)

    for i, name in enumerate(self.labels_list):
      class_roc_auc = AUROC()(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):
    """
    create AdamW optimizer with linear lr
    """
    optimizer = AdamW(self.parameters(), lr=2e-5)
    
    #scheduler = get_linear_schedule_with_warmup(
    #  optimizer,
    #  num_warmup_steps=self.n_warmup_steps,
    #  num_training_steps=self.n_training_steps
    #)

    scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
    
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )


"""
Lightning module for NN model
"""
class IntentNNModel(pl.LightningModule):
    
  def __init__(self, n_classes: int, input_size, hidden_dims, labels_list, class_weights=None, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.nn_model = NNModel(input_size, hidden_dims, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.labels_list = labels_list
    self.save_hyperparameters()
    self.criterion = WeightedBCELoss()
    self.class_weights = class_weights

  def forward(self, input_features, labels=None):
    output = self.nn_model(input_features)
    output = torch.sigmoid(output)
    loss = 0
    
    if labels is not None:
        loss = self.criterion(output, labels.float(), self.class_weights)
        #loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    #input_ids = batch[0]
    labels = batch[1]
    input_features = batch[3]
    loss, outputs = self(input_features, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    labels = batch[1]
    input_features = batch[3]
    loss, outputs = self(input_features, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    labels = batch[1]
    input_features = batch[3]
    loss, outputs = self(input_features, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()

    predictions = torch.stack(predictions)

    for i, name in enumerate(self.labels_list):
      class_roc_auc = AUROC()(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    
    #scheduler = get_linear_schedule_with_warmup(
    #  optimizer,
    #  num_warmup_steps=self.n_warmup_steps,
    #  num_training_steps=self.n_training_steps
    #)

    scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
    
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

"""
Lighting data module to load train, val and test dataset
"""
class IntentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, val_df, test_df, label_columns, tokenizer, vectorizer=None, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.label_columns = label_columns
    self.vectorizer = vectorizer
    
    #self.vocab = train_vocab
    
  def setup(self, stage=None):
    #lbl_enc = preprocessing.LabelEncoder()
    #y = lbl_enc.fit_transform(self.train_df.intent.values)

    self.train_dataset = IntentDataset(
      self.train_df,
      label_columns=self.label_columns,
      #train_vocab=self.vocab,
      tokenizer=self.tokenizer,
      max_token_len=self.max_token_len,
      vectorizer = self.vectorizer
    )

    self.val_dataset = IntentDataset(
      self.val_df,
      label_columns=self.label_columns,
      #train_vocab=self.vocab,
      tokenizer=self.tokenizer,
      max_token_len=self.max_token_len,
      vectorizer = self.vectorizer
    )
    
    #y_test = lbl_enc.transform(self.test_df.intent.values)
    self.test_dataset = IntentDataset(
      self.test_df,
      label_columns=self.label_columns,
      #train_vocab=self.vocab,
      tokenizer=self.tokenizer,
      max_token_len=self.max_token_len,
      vectorizer = self.vectorizer
    )
    
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=0,
      collate_fn=collate_batch
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=0,
      collate_fn=collate_batch
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0,
      collate_fn=collate_batch
    )


