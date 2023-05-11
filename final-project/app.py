import gradio as gr
import torch

import numpy as np
import plotly.graph_objects as go

from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification

from .hdc_predict import HDCPredictor, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

deeplearning_model_dir = "/home/satyam/hpml-final-project/final-project/tex/analysis/deeplearning_model_save/210032"
deeplearning_model = DistilBertForSequenceClassification.from_pretrained(deeplearning_model_dir)
deeplearning_model.to(device)
deeplearning_model.eval()

hdc_dir = '/home/tm3229/hpml-final-project/final-project/tex/analysis/hdc'
hdc_predictor = HDCPredictor(hdc_dir, device)

EuropeanLanguagesIndexed = [
    "Bulgarian",
    "Czech",
    "Danish",
    "Dutch",
    "German",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "Greek",
    "Hungarian",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
]


def deeplearning_predict_class(input_text):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    input_tensor = tokenizer.encode(input_text, return_tensors="pt",
                                    padding=False,
                                    truncation=True,
                                    return_attention_mask=False,
                                    return_token_type_ids=False).to(device)
    
    with torch.no_grad():
        output = deeplearning_model(input_ids=input_tensor)
        probabilities = torch.nn.functional.softmax(output.logits, dim=1).detach().cpu().numpy()[0]

    return probabilities

def hdc_predict_class(input_text):
    prediction = hdc_predictor.predict(input_text)
    probabilities = torch.nn.functional.softmax(prediction, dim=1).detach().cpu().numpy()[0]
    return probabilities


def predict_class(input_text):
    hdc_output = hdc_predict_class(input_text)
    deeplearning_output = deeplearning_predict_class(input_text)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=EuropeanLanguagesIndexed,
        y=hdc_output,
        name='HDC'
    ))

    fig.add_trace(go.Bar(
        x=EuropeanLanguagesIndexed,
        y=deeplearning_output,
        name='Deep Learning'
    ))

    fig.update_layout(barmode='group')

    # get the predicted classes
    hdc_predicted_class = EuropeanLanguagesIndexed[np.argmax(hdc_output)]
    deeplearning_predicted_class = EuropeanLanguagesIndexed[np.argmax(deeplearning_output)]

    return hdc_predicted_class, deeplearning_predicted_class, fig


output = gr.Plot()

iface = gr.Interface(
    fn=predict_class, 
    inputs=gr.inputs.Textbox(lines=2, placeholder='Enter Text...'),
    outputs=[gr.outputs.Textbox(label="HDC Predicted Language"),
        gr.outputs.Textbox(label="Deep Learning Predicted Language"),
        gr.Plot()],
    allow_flagging=False,
    title="Language Prediction",
    description="Enter a text and predict its language."
)

# Launch the app
iface.launch()