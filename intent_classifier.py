# -*- coding: utf-8 -*-

import json

import torch
import numpy as np
import pickle
from lightning_modules import IntentLSTMModel, IntentNNModel


class IntentClassifier:
    def __init__(self):
        self.model = None
        self.labels_list = []
        pass

    def is_ready(self):
        """Function to check if model is ready for inference"""
        try:
            if self.model is not None:
                self.predict("find me a flight that flies from memphis to tacoma")
                return True
        except:
            pass
        return False

    def prepare_input(self, text):
        """Function to prepare text input.
        Converts to tokenids for lstm model
        Converts to tfidf feature vector for nn model
        Parameters
        ----------
        text: str
            text input from user

        Returns
        -------
        Tuple of 
        model_input: torch.Tensor
            prepared input for model
        seq_lens: torch.Tensor
            input sequence size for lstm model
        """
        seq_lens = None
        if self.model_type == 'lstm':
            model_input = self.tokenizer.text_to_tokenids(text)
            seq_lens = torch.from_numpy(np.array([len(model_input)], dtype=np.int64))
            model_input = torch.tensor([model_input], dtype=torch.int64)
        else:
            model_input = self.vectorizer.vectorize(text)
            model_input = torch.from_numpy(np.array([model_input])).to(torch.float32)

        return model_input, seq_lens

    def predict(self, text):
        """ Predict intent for text input
        Parameters
        ----------
        text: str
            text input from user

        Returns
        -------
        predictions: dict
        """
        model_input, seq_lens = self.prepare_input(text)
        if self.model_type == 'lstm':
            _, prediction = self.model(model_input, seq_lens)
        else:
            _, prediction = self.model(model_input)

        prediction = prediction.squeeze(0).tolist()
        prediction = list(enumerate(prediction))
        prediction.sort(key=lambda x:x[1], reverse=True)
        prediction = [{
                        'label':self.labels_list[ind],
                        "confidence":prob
                        } for ind,prob in prediction[:3]]
        return {'intents':prediction}


    def load(self, file_path):
        """Load model from path 
        Parameters
        ----------
        file_path: str
            path to model data
        """
        with open(file_path, 'rb') as modelfile:
            model_data = pickle.load(modelfile)

        self.model_type = model_data['model_type']
        self.tokenizer = model_data['tokenizer']
        self.max_token_count = model_data['max_token_count']
        self.vectorizer = model_data['vectorizer']
        self.labels_list = model_data['labels_list']

        if self.model_type == 'lstm':
            self.model = IntentLSTMModel.load_from_checkpoint(
                model_data['model_path']
            )
        else:
            self.model = IntentNNModel.load_from_checkpoint(
              model_data['model_path']
            )
        self.model.eval()
        self.model.freeze()


if __name__ == '__main__':
    cls = IntentClassifier()
    print(cls.is_ready())
    cls.load('model_nn.pkl')
    print(cls.predict("find me a flight that flies from memphis to tacoma"))
    print(cls.is_ready())
    pass
