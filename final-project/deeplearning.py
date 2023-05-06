import logging
import os

import torch
import torchmetrics
from tqdm import tqdm
import torch.utils.data as data
from transformers import DistilBertForSequenceClassification, AdamW

from . import data
from .args import parse_args

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using {} device".format(args.device))

    # get the data
    train_loader, val_loader, test_loader = data.get_eurolang(**vars(args))
    num_classes = len(train_loader.dataset.classes)

    output_dir = './model_save/deeplearning/'
    model = None

    if not os.path.exists(output_dir):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
        model.to(args.device)
        model.train()

        optim = AdamW(model.parameters(), lr=5e-5)

        # train the model
        for batch in tqdm(train_loader, desc="Training"):
            samples = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            optim.zero_grad()
            outputs = model(samples, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

        # save the model    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)
        model.save_pretrained(output_dir)
    else:
        # retrieve saved model
        logging.info("Using saved model from {}".format(output_dir))
        model = DistilBertForSequenceClassification.from_pretrained(output_dir)
        print("Loaded the existing model from", output_dir)

    # prepare for testing
    model.eval()
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(args.device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            samples = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(samples)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy(predictions, labels)

    acc = accuracy.compute().cpu().item()
    print(f"Testing accuracy of {(acc * 100):.3f}%")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    