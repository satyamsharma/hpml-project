"""This module implements the hyperdimensional computing (HDC) experiments.

The experiments (for each dataset) are as follows:
- Accuracy, precision, recall, and F1-score with respect to dataset size
- Training and inference time
- FLOP analysis
- Robustness to text data corruption

Examples:
    $ python -m final-project.hdc \
        --dataset eurolang \
        --subset 0.1 \
        --roll-matrix
"""
from contextlib import nullcontext
import logging
import os
import pickle

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
from thop import profile

from . import data
from .args import parse_args


class Encoder(nn.Module):

    def __init__(self, out_features, size, roll_matrix=False):
        super(Encoder, self).__init__()
        # create the lookup table for the alphabet a-z, space and padding
        # mapping each character id to a random vector of size out_features
        self.symbol = embeddings.Random(size,
                                        out_features,
                                        padding_idx=data.PADDING_IDX)
        # if roll_matrix, implement the roll operation for a roll of 1, 2
        # as a matrix multiplication
        self.roll_matrix = roll_matrix
        if roll_matrix:
            roll_one = self.create_roll_matrix(3, 1)
            roll_two = self.create_roll_matrix(3, 2)
            self.register_buffer('roll_one', roll_one)
            self.register_buffer('roll_two', roll_two)

    @staticmethod
    def create_roll_matrix(features, shift):
        """Create a roll matrix for a given shift."""
        eye = torch.eye(features)
        return torch.roll(eye, shifts=-shift, dims=0)

    def forward(self, x):
        symbols = self.symbol(x)
        if self.roll_matrix:
            # roll_two all rows up to the last two rows
            symbols_two = torch.matmul(symbols[:, 0:-2, -3:], self.roll_two)
            # roll_one the first row up to the last row
            symbols_one = torch.matmul(symbols[:, 1:-1, -3:], self.roll_one)
            # third character doesn't require a roll
            symbols_zero = symbols[:, 2:, -3:]
            # element-wise multiply the three rolls
            symbols_rolled = symbols_zero * symbols_one * symbols_two
            symbols_mult = symbols[:, 0:-2, :] * symbols[:, 1:-1, :] * symbols[:, 2:, :]
            # set the elements in symbols_mult to symbols_rolled
            symbols_mult[:, :, -3:] = symbols_rolled
            # row-wise sum
            sample_hv = torch.sum(symbols_mult, dim=1)
        else:
            sample_hv = torchhd.ngrams(symbols, n=3)
        return torchhd.hard_quantize(sample_hv)

def test_model(model, encode, test_loader, device):
    final_outputs = []
    complete_labels = []
    with torch.no_grad():
        model.normalize()
        model.eval()
        test_start = torch.cuda.Event(enable_timing=True)
        test_end = torch.cuda.Event(enable_timing=True)
        test_start.record()
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)
            labels = labels.to(device)
            complete_labels.append(labels)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            final_outputs.append(outputs)
        test_end.record()

        torch.cuda.synchronize()
        testing_time = test_start.elapsed_time(test_end) / 1000
        logging.info(f"Testing time: {testing_time:.3f}s")
    return final_outputs, complete_labels, testing_time
    
def run_model(train_loader,
              test_loader,
              args, record_speed=False):
    """Run the HDC model."""
    encode = Encoder(data.DIMENSIONS, data.NUM_TOKENS, roll_matrix=args.roll_matrix)
    encode = encode.to(args.device)

    num_classes = len(train_loader.dataset.classes)
    model = Centroid(data.DIMENSIONS, num_classes)
    model = model.to(args.device)

    profiler = None
    if args.profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=args.log_dir),
            record_shapes=True,
            profile_memory=True,
        )

    with profiler if profiler else nullcontext():
        train_start = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)
        train_start.record()
        with torch.no_grad():
            for samples, labels in tqdm(train_loader, desc="Training"):
                samples = samples.to(args.device)
                labels = labels.to(args.device)

                samples_hv = encode(samples)
                model.add(samples_hv, labels)

                if profiler:
                    profiler.step()
        train_end.record()

        torch.cuda.synchronize()
        training_time = train_start.elapsed_time(train_end) / 1000
        logging.info(f"Training time: {training_time:.3f}s")
    
    if profiler:
        # profiler.export_chrome_trace(os.path.join(log_dir, 'trace.json'))
        logging.info(profiler.key_averages().table(sort_by='cuda_time_total',
                                                   row_limit=10))

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    acc = None
    if test_loader is not None:
        final_outputs, complete_labels, testing_time = test_model(model, encode, test_loader, args.device)
        for outputs, labels in zip(final_outputs, complete_labels):
            # TODO: understand what's happening under the hood here
            # TODO: determine if accuracy can be run on the GPU
            accuracy.update(outputs.cpu(), labels.cpu())
        acc = accuracy.compute().item()
        logging.info(f"Testing accuracy of {(acc * 100):.3f}%")
        
    # record times
    if args.output_dir is not None and record_speed:
        speed_analysis_filepath = os.path.join(args.output_dir, 'speed_analysis.csv')
        if os.path.exists(speed_analysis_filepath):
            df = pd.read_csv(speed_analysis_filepath)
        else:
            df = pd.DataFrame(columns=['Model', 'Training-Time', 'Testing-Time'])
        temp_df = pd.DataFrame([['HDC', training_time, testing_time]], columns=['Model', 'Training-Time', 'Testing-Time'])
        # overwrite HDC row if it exists
        if 'HDC' in df['Model'].values:
            df.loc[df['Model'] == 'HDC'] = temp_df.iloc[:1]
        else:
            df = pd.concat((df, temp_df))
        df = df.sort_values(by=['Model'])
        # create output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(speed_analysis_filepath, index=False)
        # save as LaTeX table too, center columns
        df.to_latex(os.path.join(args.output_dir, 'speed_analysis.tex'), index=False, column_format='c' * len(df.columns))
    return acc, model, encode



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
            logging.info(
                f'Skipping {size * 100}% data size experiment; already ran.')
            continue
        args.subset = size
        train_loader, _, test_loader = data.get_eurolang(**vars(args))
        example_count = int(len(train_loader.dataset))
        logging.info(
            f'Training with {size * 100}% of the dataset ({example_count} examples)'
        )

        # run the model
        acc, _, _ = run_model(train_loader, test_loader, args.device)
        temp_df = pd.DataFrame(
            [[example_count, size, acc]],
            columns=['Examples', 'Dataset Pct.', 'Accuracy'])
        df = df.append(temp_df)
    df = df.sort_values(by=['Dataset Pct.'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'hdc_data_size.csv'), index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'hdc_data_size.tex'),
                index=False,
                float_format="%.4f",
                column_format='c' * len(df.columns))


def flop_analysis(args):
    """Flop analysis for single sample inference on model trained with full training set"""
    logging.info('Running flop analysis experiment...')
    flop_analysis_filepath = os.path.join(args.output_dir, 'flop_analysis.csv')
    if args.use_cache and os.path.exists(flop_analysis_filepath):
        logging.info('Using cached data for flop analysis experiment...')
        df = pd.read_csv(flop_analysis_filepath)
        if 'HDC' in df['Model'].values:
            logging.info('Skipping flop analysis experiment; already ran.')
            logging.info(df)
            return

    train_loader, _, _ = data.get_eurolang(**vars(args))
    _, model, encode = run_model(train_loader, None, args.device, enable_profiler=args.profiler, log_dir=args.log_dir)
    model.eval()
    text = "The quick brown fox jumps over the lazy dog "
    prepared_input = data.prepare_input_sentence(text)
    encoded_sentence = data.transform(prepared_input).unsqueeze(0).to(args.device)

    macs = 0
    params = 0
    flops = 0
    with torch.no_grad():
        macs_, params_ = profile(encode, inputs=(encoded_sentence,), verbose=False)
        flops_ = macs_ * 2
        logging.info(f"Encoding FLOPs: {flops_}",)
        logging.info(f"Encoding Params: {params_}")
        macs += macs_
        params += params_
        flops += flops_

        samples_hv = encode(encoded_sentence)
        macs_, params_ = profile(model, inputs=(samples_hv,), verbose=False)
        flops_ = macs_ * 2
        logging.info(f"Model FLOPs: {flops_}")
        logging.info(f"Model Params: {params_}")
        macs += macs_
        params += params_
        flops += flops_

    # args.subset = temp_subset
    if os.path.exists(flop_analysis_filepath):
        df = pd.read_csv(flop_analysis_filepath)
    else:
        df = pd.DataFrame(columns=['Model', 'Parameters', 'FLOPs'])
    
    temp_df = pd.DataFrame([['HDC', params, flops]], columns=['Model', 'Parameters', 'FLOPs'])
    temp_df['Parameters'] = temp_df['Parameters'].astype('int')
    temp_df['FLOPs'] = temp_df['FLOPs'].astype('int')
    # overwrite HDC row if it exists
    if 'HDC' in df['Model'].values:
        df.loc[df['Model'] == 'HDC'] = temp_df.iloc[:1]
    else:
        df = pd.concat((df, temp_df))
    df = df.sort_values(by=['Model'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(flop_analysis_filepath, index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'flop_analysis.tex'), index=False, column_format='c' * len(df.columns))


def load_model(centroid_filepath, encoder_filepath):
    # reload the models and verify accuracy
    with open(centroid_filepath, 'rb') as f:
        centroid_loaded = pickle.load(f)
    with open(encoder_filepath, 'rb') as f:
        encoder_loaded = pickle.load(f)
    return centroid_loaded, encoder_loaded

def save_model(centroid, encoder, acc, args):
    # save the centroid and encoder
    hdc_model_dir = os.path.join(args.output_dir, 'hdc')
    os.makedirs(hdc_model_dir, exist_ok=True)
    centroid_filepath = os.path.join(hdc_model_dir, 'centroid.pkl')
    encoder_filepath = os.path.join(hdc_model_dir, 'encoder.pkl')
    logging.info(f'Saving HDC model to {hdc_model_dir}...')
    with open(centroid_filepath, 'wb') as f:
        pickle.dump(centroid, f)
    with open(encoder_filepath, 'wb') as f:
        pickle.dump(encoder, f)
    
    centroid_loaded, encoder_loaded = load_model(centroid_filepath, encoder_filepath)
    _, _, test_loader = data.get_eurolang(**vars(args))
    final_outputs, complete_labels, testing_time = test_model(centroid_loaded, encoder_loaded, test_loader, args.device)
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(test_loader.dataset.classes))
    for outputs, labels in zip(final_outputs, complete_labels):
        # TODO: understand what's happening under the hood here
        # TODO: determine if accuracy can be run on the GPU
        accuracy.update(outputs.cpu(), labels.cpu())
    acc_loaded = accuracy.compute().item()
    logging.info(f'Accuracy of loaded model: {acc_loaded}')
    assert acc == acc_loaded, 'Loaded model accuracy does not match original model accuracy'

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using {} device".format(args.device))
    # get the data
    train_loader, _, test_loader = data.get_eurolang(**vars(args))

    if args.experiment == 'baseline' or args.experiment == 'all':
        acc, centroid, encoder = run_model(train_loader, test_loader, args)
        if args.save_model:
            save_model(centroid, encoder, acc, args)
    elif args.experiment == 'data-size' or args.experiment == 'all':
        data_size_experiment(args)
    elif args.experiment == 'flop' or args.experiment == 'all':
        flop_analysis(args)
    elif args.experiment == 'speed' or args.experiment == 'all':
        run_model(train_loader, test_loader, args, record_speed=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
