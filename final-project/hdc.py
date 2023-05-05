"""This module implements the hyperdimensional computing (HDC) experiments.

The experiments (for each dataset) are as follows:
- Accuracy, precision, recall, and F1-score with respect to dataset size
- Training and inference time
- FLOP analysis
- Robustness to text data corruption
"""
import logging
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets import EuropeanLanguages as Languages

from . import data
from .args import parse_args

class Encoder(nn.Module):
    def __init__(self, out_features, size):
        super(Encoder, self).__init__()
        # create the lookup table for the alphabet a-z, space and padding
        # mapping each character id to a random vector of size out_features
        self.symbol = embeddings.Random(size, out_features, padding_idx=data.PADDING_IDX)

    def forward(self, x):
        symbols = self.symbol(x)
        sample_hv = torchhd.ngrams(symbols, n=3)
        return torchhd.hard_quantize(sample_hv)


def run_model(train_loader, test_loader, device):
    """Run the HDC model."""
    encode = Encoder(data.DIMENSIONS, data.NUM_TOKENS)
    encode = encode.to(device)

    num_classes = len(train_loader.dataset.classes)
    model = Centroid(data.DIMENSIONS, num_classes)
    model = model.to(device)

    with torch.no_grad():
        for samples, labels in tqdm(train_loader, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        model.normalize()

        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            # TODO: understand what's happening under the hood here
            # TODO: determine if accuracy can be run on the GPU
            accuracy.update(outputs.cpu(), labels.cpu())
    acc = accuracy.compute().item()
    print(f"Testing accuracy of {(acc * 100):.3f}%")
    return acc

def data_size_experiment(args):
    """Vary the dataset size HDC model."""
    logging.info('Running data size experiment...')
    hdc_data_size_filepath = os.path.join(args.output_dir, 'hdc_data_size.csv')
    if args.use_cache and os.path.exists(hdc_data_size_filepath):
        logging.info('Using cached data for HDC data size experiment...')
        df = pd.read_csv(hdc_data_size_filepath)
    else:
        df = pd.DataFrame(columns=['Examples', 'Dataset Pct.', 'Accuracy'])
    sizes = [0.0001, 0.001, 0.01, 0.02, 0.05]
    for size in sizes:
        if size in df['Dataset Pct.'].values:
            logging.info(f'Skipping {size * 100}% data size experiment; already ran.')
            continue
        args.subset = size
        train_loader, _, test_loader = data.get_eurolang(**vars(args))
        example_count = int(len(train_loader.dataset))
        logging.info(f'Training with {size * 100}% of the dataset ({example_count} examples)')

        # run the model
        acc = run_model(train_loader, test_loader, args.device)
        temp_df = pd.DataFrame([[example_count, size, acc]], columns=['Examples', 'Dataset Pct.', 'Accuracy'])
        df = df.append(temp_df)
    df = df.sort_values(by=['Dataset Pct.'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'hdc_data_size.csv'), index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'hdc_data_size.tex'), index=False, float_format="%.4f", column_format='c' * len(df.columns))


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using {} device".format(args.device))

    # get the data
    train_loader, _, test_loader = data.get_eurolang(**vars(args))

    if args.experiment == 'baseline' or args.experiment == 'all':
        run_model(train_loader, test_loader, args.device)
    elif args.experiment == 'data-size' or args.experiment == 'all':
        data_size_experiment(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    