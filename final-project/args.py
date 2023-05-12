"""This module stores all arguments for the different modules."""
import argparse
import os
import sys
import logging
from . import logconfig  # import default logger setup
import random

import torch
import numpy as np


def parse_args(args=None):
    """Parse command line arguments."""
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Load and process data.')
    parser.add_argument('--dataset',
                        type=str,
                        default='eurolang',
                        help='Dataset to load and process.',
                        choices=['eurolang'])
    parser.add_argument('--data-dir',
                        type=os.path.expanduser,
                        default='~/data')
    parser.add_argument(
        '--output-dir',
        type=os.path.expanduser,
        default='~/hpml-final-project/final-project/tex/analysis')
    parser.add_argument(
        '--log-dir',
        type=os.path.expanduser,
        default='~/hpml-final-project/logs',
        help='Directory to store logs such as profiler output.')
    parser.add_argument('--subset',
                        type=float,
                        default=1.0,
                        help='Percentage of training dataset to use.')
    parser.add_argument(
        '--train-size',
        type=float,
        default=1.0,
        help=
        'Percentage of training dataset to use with the remainder used as a validation set. If subset is specified, then the training dataset is first subsetted. Then train-size percent of the subsetted data is used as the training set and the remainder is used for a validation set.'
    )
    parser.add_argument(
        '--corruption-rate',
        type=float,
        default=0.0,
        help='Corruption rate for the training data sentences.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed to use for subsetting the dataset.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size.')
    parser.add_argument('--roll-matrix',
                        action='store_true',
                        default=False,
                        help='Whether to use a roll matrix for the encoder.')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
    )
    # TODO: add option for HDC
    parser.add_argument(
        '--deep-learning',
        action='store_true',
        default=False,
        help='Whether to prepare the data for a deep learning model.')
    parser.add_argument('--tokenizer',
                        type=str,
                        default='distilbert-base-uncased',
                        help='Tokenizer to use for the deep learning model.')
    parser.add_argument('--profiler',
                        action='store_true',
                        default=False,
                        help='Whether to run the profiler.')
    parser.add_argument('--experiment',
                        type=str,
                        default='baseline',
                        help='Which experiment to run.',
                        choices=[
                            'baseline', 'data-size', 'speed', 'flop',
                            'corruption', 'all'
                        ])
    parser.add_argument(
        '--use-cache',
        action='store_true',
        default=False,
        help=
        'Whether to use cached experiment run data to speed up subsequent experiments or finetune table formatting.'
    )
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='Whether to save the model.')
    args = parser.parse_args(args)
    # set log level based on command line argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    logging.getLogger().setLevel(numeric_level)

    logging.info(f'Running with:\n {args}')

    # TODO: need to make sure these seeds are set in other modules that call get_eurolang
    # set random seed
    logging.info(f'Setting random seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args