# HPML Final Project
Todd Morrill (tm3229@columbia.edu), Satyam Sharma (ss6522@columbia.edu)

## Language Identification
**Experiment description:** We will develop deep learning and hyperdimensional computing (HDC) based classifiers to identify languages using the [European languages](https://torchhd.readthedocs.io/en/stable/datasets.html#torchhd.datasets.EuropeanLanguages)
dataset and compare them on several key dimensions, namely:
1. Accuracy, precision, recall, and F1-score with respect to dataset size
1. Training and inference time
1. FLOP analysis
1. and robustness to text data corruption

### Deep Learning Approach
We will use [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) as our modeling backbone and largely follow this [guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) to implement the classifier.


### Hyperdimensional Computing Approach
We will follow the approach of [this paper](https://iis-people.ee.ethz.ch/~arahimi/papers/ISLPED16.pdf) and use this reference implementation in [TorchHD](https://github.com/hyperdimensional-computing/torchhd/blob/main/examples/language_recognition.py).
