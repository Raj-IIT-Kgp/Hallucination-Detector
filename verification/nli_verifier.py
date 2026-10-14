import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import threading


model_name = "facebook/bart-large-mnli"


device = torch.device(os.environ.get("NLI_DEVICE", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name
).to(device)

# Global lock: PyTorch BART model and HuggingFace tokenizer use C-level memory
# that is NOT safe to call from multiple threads simultaneously on macOS
# (causes "double free" / malloc corruption). Serialise all inference calls.
_inference_lock = threading.Lock()


labels = [
    "contradicted",
    "unknown",
    "supported"
]


def verify_claim(claim, evidence, domain="general"):

    with _inference_lock:
        inputs = tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[0][prediction].item()
        prob_list = [probabilities[0][i].item() for i in range(len(labels))]

        # Explicitly synchronize GPU/MPS command buffer before releasing lock
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    label = labels[prediction]

    # Dynamic confidence thresholding
    threshold = 0.55 if domain == "scientific" else 0.75

    # If the model is uncertain, default to unknown to prevent false positives
    if confidence < threshold:
        label = "unknown"

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_list
    }


def verify_claims_batch(claim, evidence_texts, domain="general"):
    """
    Verifies multiple evidence snippets against a single claim in a SINGLE batched forward pass on GPU/MPS.
    Eliminates sequential loop overhead, delivering a massive speedup on snippet evaluation.
    """
    if not evidence_texts:
        return []

    threshold = 0.55 if domain == "scientific" else 0.75

    with _inference_lock:
        inputs = tokenizer(
            evidence_texts,
            [claim] * len(evidence_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predictions = torch.argmax(probabilities, dim=1).tolist()

        results = []
        for i, pred in enumerate(predictions):
            conf = probabilities[i][pred].item()
            prob_list = [probabilities[i][c].item() for c in range(len(labels))]
            lbl = labels[pred]
            if conf < threshold:
                lbl = "unknown"
            results.append({
                "label": lbl,
                "confidence": conf,
                "probabilities": prob_list
            })

        # Explicitly synchronize GPU/MPS command buffer before releasing lock
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    return results