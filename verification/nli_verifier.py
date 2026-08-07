from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "facebook/bart-large-mnli"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)


labels = [
    "contradicted",
    "unknown",
    "supported"
]


def verify_claim(claim, evidence):

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True
    )


    with torch.no_grad():

        outputs = model(**inputs)


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    prediction = torch.argmax(
        probabilities
    ).item()


    return {
        "label": labels[prediction],
        "confidence": probabilities[0][prediction].item()
    }