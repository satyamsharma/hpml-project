"""This module loads a trained model and uses it for prediction on an arbitrary
input sentence.

Examples:
    $ python -m final-project.hdc_predict
        
"""
import os
import logging

import torch
import torchmetrics
from torchhd.models import Centroid
from . import data
from .hdc import Encoder, load_model, test_model
from .args import parse_args

class HDCPredictor(object):
    def __init__(self, model_dir, device='cpu'):
        centroid_filepath = os.path.join(model_dir, 'centroid.pkl')
        encoder_filepath = os.path.join(model_dir, 'encoder.pkl')
        self.centroid, self.encoder = load_model(centroid_filepath, encoder_filepath)
        self.centroid.eval()
        self.encoder.eval()
        self.centroid.to(device)
        self.encoder.to(device)
        self.device = device

    def predict(self, sentence):
        prepared_input = data.prepare_input_sentence(sentence)
        encoded_sentence = data.transform(prepared_input).unsqueeze(0).to(self.device)
        encoded_sentence = self.encoder(encoded_sentence)
        prediction = self.centroid(encoded_sentence)
        return prediction

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(args.output_dir, 'hdc')
    predictor = HDCPredictor(model_dir, args.device)
    english_sentence = 'The quick brown fox jumps over the lazy dog.'
    french_sentence = 'Le renard brun rapide saute par-dessus le chien paresseux.'
    italian_sentence = 'La volpe marrone veloce salta sopra il cane pigro.'
    english_corrupted = 'teh quicgk brown fox jumps ovrr hte lay dog'
    for sentence in [english_sentence, french_sentence, italian_sentence, english_corrupted]:
        logging.info(f'Predicting sentence: {sentence}')
        prediction = predictor.predict(sentence)
        probabilities = torch.nn.functional.softmax(prediction, dim=1).detach().cpu().numpy()[0]
        logging.info(f'The predicted class is {data.CLASS_LABELS[prediction.argmax(dim=1).item()]}')
    
    # verify test set accuracy
    test_loader = data.get_eurolang(**vars(args))[2]
    acc, testing_time = test_model(predictor.centroid, predictor.encoder, test_loader, args.device)
    logging.info(f'Accuracy of loaded model: {100 * acc:.3f}%')

if __name__ == '__main__':
    args = parse_args()
    main(args)
