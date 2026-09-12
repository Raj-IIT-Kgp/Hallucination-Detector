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


def verify_claim(claim, evidence, domain="general"):

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


    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[0][prediction].item()
    
    label = labels[prediction]
    
    # Dynamic confidence thresholding
    threshold = 0.55 if domain == "scientific" else 0.75
    
    # If the model is uncertain, default to unknown to prevent false positives
    if confidence < threshold:
        label = "unknown"

    return {
        "label": label,
        "confidence": confidence
    }