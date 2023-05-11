"""This module contains functions for loading and processing data.

In particular, we want to be able to specify a data set and any necessary configurations for that dataset. For example, this module can load the European Languages dataset and return train and test Torch data loaders for that dataset, subsetted by a specified percentage. This module should also implement the text corruption functions.

Examples:
    $ python -m final-project.data \
        --dataset eurolang \
        --subset 0.1

    $ python -m final-project.data \
        --dataset eurolang \
        --subset 0.1 \
        --corruption-rate 0.1
"""
import logging
from . import logconfig

from transformers import DataCollatorWithPadding, AutoTokenizer
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchhd.datasets import EuropeanLanguages as Languages
from tqdm import tqdm

from .args import parse_args

CLASS_LABELS = ['Bulgarian', 'Czech', 'Danish', 'Dutch', 'German', 'English', 'Estonian', 'Finnish', 'French', 'Greek', 'Hungarian', 'Italian', 'Latvian', 'Lithuanian', 'Polish', 'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Spanish', 'Swedish']
DIMENSIONS = 10000
# cap maximum sample size to 128 characters (including spaces)
MAX_INPUT_SIZE = 128
PADDING_IDX = 0

ASCII_A = ord("a")
ASCII_Z = ord("z")
ASCII_SPACE = ord(" ")
NUM_TOKENS = ASCII_Z - ASCII_A + 3  # a through z plus space and padding

def corrupt_sentence(sentence, corruption_rate):
    """Corrupts a sentence by randomly inserting, deleting, swapping, or replacing
    letters from the set of lowercased letters [a-z]. A position will be
    corrupted with probability corruption rate, and then we decide what type of
    corruption to apply with 1/3 probability to each action."""
    i = 0
    while i < len(sentence):
        if random.random() < corruption_rate:
            # decide what type of corruption to apply
            corruption_type = random.randint(0, 3)
            if corruption_type == 0:
                # insert a random character
                sentence = sentence[:i] + chr(random.randint(ASCII_A, ASCII_Z)) + sentence[i:]
            elif corruption_type == 1:
                # delete the current character
                sentence = sentence[:i] + sentence[i+1:]
            elif corruption_type == 2:
                # swap the current character with the next character
                if i < len(sentence) - 1:
                    sentence = sentence[:i] + sentence[i+1] + sentence[i] + sentence[i+2:]
            else:
                # replace the current character with a random character
                sentence = sentence[:i] + chr(random.randint(ASCII_A, ASCII_Z)) + sentence[i+1:]
        # increment i
        i += 1
    return sentence

def prepare_input_sentence(sentence):
    """Lowercases the sentence and removes any characters that are not in [a-z] or space."""
    sentence = sentence.lower()
    sentence = ''.join([c for c in sentence if ((c >= 'a' and c <= 'z') or c == ' ')])
    return sentence


def char2int(char: str) -> int:
    """Map a character to its integer identifier"""
    ascii_index = ord(char)

    if ascii_index == ASCII_SPACE:
        # Remap the space character to come after "z"
        return ASCII_Z - ASCII_A + 1

    return ascii_index - ASCII_A


def transform(x: str) -> torch.Tensor:
    char_ids = x[:MAX_INPUT_SIZE]
    char_ids = [char2int(char) + 1 for char in char_ids.lower()]

    if len(char_ids) < MAX_INPUT_SIZE:
        char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

    return torch.tensor(char_ids, dtype=torch.long)


class LanguagesDL(data.Dataset):
    """Dataset for the European Languages dataset for the deep learning model."""

    def __init__(self, dataset, tokenizer):
        super().__init__()
        # pull in attributes from the TorchHD Languages dataset
        self.data = dataset.data
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.files = dataset.files
        self.tokenizer = tokenizer
        logging.debug(f'List of classes:\n{self.classes}')

        # tokenize the data, handle padding in collate_fn
        self.encoded_data = self.tokenizer(
            self.data,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False)['input_ids']

    def __getitem__(self, index):
        # retrieve input_ids
        text_sample = self.encoded_data[index]
        label = self.targets[index]
        return {'input_ids': text_sample, 'label': label}

    def __len__(self):
        return len(self.data)


def get_eurolang(data_dir,
                 subset,
                 train_size,
                 batch_size,
                 deep_learning=True,
                 tokenizer='distilbert-base-uncased',
                 **kwargs):
    # TODO: consider how to recycle this function for an inference dataset with no labels
    logging.info(f'Caching data in {data_dir}')
    # saves to/loads from args.data_dir/language-recognition
    train_ds = Languages(data_dir,
                         train=True,
                         transform=transform,
                         download=True)
    test_ds = Languages(data_dir,
                        train=False,
                        transform=transform,
                        download=True)
    logging.debug(
        f'Loaded {len(train_ds)} training samples and {len(test_ds)} testing samples.'
    )

    # if deep_learning, need to process the text differently and use a collate function
    collate_fn = None
    if deep_learning:
        logging.info('Preparing data for the deep learning model.')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        train_ds = LanguagesDL(train_ds, tokenizer=tokenizer)
        test_ds = LanguagesDL(test_ds, tokenizer=tokenizer)
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # subset set dataset, if specified, for rapid development
    if (subset is not None) and (subset < 1.0):
        logging.info(
            f'Subsetting training dataset to {subset*100:.0f}% of original size.'
        )
        # select subset percent of indices
        train_indices = random.sample(range(len(train_ds)),
                                      int(len(train_ds) * subset))
        train_ds = torch.utils.data.Subset(train_ds, train_indices)

    # create a train and validation split
    # 80% train, 20% validation
    if (train_size is not None) and (train_size < 1.0):
        logging.info(
            f'Splitting training dataset into {train_size*100:.0f}% train and {100-train_size*100:.0f}% validation.'
        )
    train_size = int(train_size * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = data.random_split(train_ds, [train_size, val_size])

    # now we're missing the classes attribute, so we need to reassign it
    train_ds.classes = test_ds.classes
    val_ds.classes = test_ds.classes

    train_ld = data.DataLoader(train_ds,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=collate_fn)
    val_ld = data.DataLoader(val_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)
    test_ld = data.DataLoader(test_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)
    logging.debug(f'Train size: {len(train_ds)}')
    logging.debug(f'Validation size: {len(val_ds)}')
    logging.debug(f'Test size: {len(test_ds)}')
    return train_ld, val_ld, test_ld

def main(args):
    sentence = 'the quick brown fox jumps over the lazy dog'
    logging.info(f'Original sentence: {sentence}')
    sentence = corrupt_sentence(sentence, corruption_rate=args.corruption_rate)
    logging.info(f'Corrupted sentence: {sentence}')
    exit()
    train_loader, val_loader, test_loader = get_eurolang(**vars(args))
    logging.info('Sample batch:')
    for batch in train_loader:
        print(batch)
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)