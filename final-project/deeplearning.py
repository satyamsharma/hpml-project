import logging
import os

import torch
import torchmetrics
from tqdm import tqdm
import torch.utils.data as data
from transformers import DistilBertForSequenceClassification, AdamW

from transformers import AutoTokenizer
import pandas as pd

from fvcore.nn import FlopCountAnalysis

from . import data
from .args import parse_args

def get_trained_model(train_loader, device):
    num_classes = len(train_loader.dataset.classes)
    output_dir = args.output_dir + "/deeplearning_model_save/{}/".format(int(len(train_loader.dataset)))

    if os.path.exists(output_dir) and args.use_cache:
        # retrieve saved model
        logging.info("Using saved model from {}".format(output_dir))
        model = DistilBertForSequenceClassification.from_pretrained(output_dir)
        model.to(device)
        logging.info("Loaded the existing model from {}".format(output_dir))
    else:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
        model.to(device)
        model.train()

        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # train the model
        for batch in tqdm(train_loader, desc="Training"):
            samples = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optim.zero_grad()
            outputs = model(samples, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

        # save the model    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info("Saving model to %s" % output_dir)
        model.save_pretrained(output_dir)

    return model

def run_model(train_loader, test_loader, device):
    model = get_trained_model(train_loader, device)
    num_classes = len(train_loader.dataset.classes)

    # prepare for testing
    model.eval()
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            samples = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(samples)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy(predictions, labels)

    acc = accuracy.compute().cpu().item()
    print(f"Testing accuracy of {(acc * 100):.3f}%")
    return acc

def data_size_experiment(args):
    """Vary the dataset size DeepLearning model."""
    logging.info('Running data size experiment...')
    deeplearning_data_size_filepath = os.path.join(args.output_dir, 'deeplearning_data_size.csv')

    if args.use_cache and os.path.exists(deeplearning_data_size_filepath):
        logging.info('Using cached data for deeplearning data size experiment...')
        df = pd.read_csv(deeplearning_data_size_filepath)
    else:
        df = pd.DataFrame(columns=['Examples', 'Dataset Pct.', 'Accuracy'])
    sizes = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    temp_subset = args.subset

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
        df = df._append(temp_df)

    args.subset = temp_subset
    df = df.sort_values(by=['Dataset Pct.'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'deeplearning_data_size.csv'), index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'deeplearning_data_size.tex'), index=False, float_format="%.4f", column_format='c' * len(df.columns))

def flop_analysis(args):
    """Flop analysis for single sample inference on model trained with full training set"""
    logging.info('Running flop analysis experiment...')

    flop_analysis_filepath = os.path.join(args.output_dir, 'flop_analysis.csv')
    temp_subset = args.subset
    args.subset = 1

    train_loader, _, _ = data.get_eurolang(**vars(args))
    model = get_trained_model(train_loader, args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    text = "The quick brown fox jumps over the lazy dog "
    input_ids = tokenizer.encode(text, return_tensors="pt").to(args.device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_ids).total()
        params = sum(p.numel() for p in model.parameters())
        print("Flops: ", flops)
        print("Parameters: ", params)

    args.subset = temp_subset
    if os.path.exists(flop_analysis_filepath):
        df = pd.read_csv(flop_analysis_filepath)
    else:
        df = pd.DataFrame(columns=['Model', 'Parameters', 'FLOPs'])

    df = df[df['Model'] != 'distilbert-base-uncased']

    temp_df = pd.DataFrame([['distilbert-base-uncased', params, flops]], columns=['Model', 'Parameters', 'FLOPs'])
    temp_df['Parameters'] = temp_df['Parameters'].astype('int')
    temp_df['FLOPs'] = temp_df['FLOPs'].astype('int')
    df = df._append(temp_df)
    df = df.sort_values(by=['Model'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'flop_analysis.csv'), index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'flop_analysis.tex'), index=False, column_format='c' * len(df.columns))

def speed_analysis(args):
    """Speed analysis with full training and testing set"""
    logging.info('Running Speed analysis experiment...')
    logging.info('tqdm is disabled')

    temp_subset = args.subset

    # Warming up Training...
    args.subset = 0.01
    warmup_train_loader, _, warmup_test_loader = data.get_eurolang(**vars(args))
    num_classes = len(warmup_train_loader.dataset.classes)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    model.to(args.device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Warming up training...")
    for batch in warmup_train_loader:
        samples = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        optim.zero_grad()
        outputs = model(samples, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

    args.subset = 1
    train_loader, _, test_loader = data.get_eurolang(**vars(args))
    num_classes = len(train_loader.dataset.classes)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    model.to(args.device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # train the model
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print("Training...")
    start.record()
    for batch in train_loader:
        samples = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        optim.zero_grad()
        outputs = model(samples, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
    end.record()

    torch.cuda.synchronize()
    training_time = start.elapsed_time(end)

    print("Warming up testing...")
    with torch.no_grad():
        for batch in warmup_test_loader:
            samples = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(samples)

    print("Testing...")
    start.record()
    with torch.no_grad():
        for batch in test_loader:
            samples = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(samples)
    end.record()

    torch.cuda.synchronize()
    testing_time = start.elapsed_time(end)

    print("training_time ", training_time, " ms")
    print("testing_time ", testing_time, " ms")

    speed_analysis_filepath = os.path.join(args.output_dir, 'speed_analysis.csv')

    args.subset = temp_subset
    if os.path.exists(speed_analysis_filepath):
        df = pd.read_csv(speed_analysis_filepath)
    else:
        df = pd.DataFrame(columns=['Model', 'Training-Time', 'Testing-Time'])

    # delete old record
    df = df[df['Model'] != 'distilbert-base-uncased']

    temp_df = pd.DataFrame([['distilbert-base-uncased', training_time, testing_time]], columns=['Model', 'Training-Time', 'Testing-Time'])
    df = df._append(temp_df)
    df = df.sort_values(by=['Model'])
    # create output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'speed_analysis.csv'), index=False)
    # save as LaTeX table too, center columns
    df.to_latex(os.path.join(args.output_dir, 'speed_analysis.tex'), index=False, column_format='c' * len(df.columns))


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using {} device".format(args.device))

    # get the data
    train_loader, val_loader, test_loader = data.get_eurolang(**vars(args))

    if args.experiment == 'baseline' or args.experiment == 'all':
        run_model(train_loader, test_loader, args.device)
    elif args.experiment == 'data-size' or args.experiment == 'all':
        data_size_experiment(args)
    elif args.experiment == 'flop' or args.experiment == 'all':
        flop_analysis(args)
    elif args.experiment == 'speed' or args.experiment == 'all':
        speed_analysis(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    