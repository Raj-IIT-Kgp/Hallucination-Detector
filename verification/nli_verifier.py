import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import threading


# Override with NLI_MODEL (e.g. "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli").
# A different model needs a one-off download (HF_HUB_OFFLINE=0) and a conformal
# recalibration (experiments/eval_conformal.py), since q_hat was fit on BART posteriors.
model_name = os.environ.get("NLI_MODEL", "facebook/bart-large-mnli")


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

# Models disagree on logit order (BART: contra/neutral/entail, DeBERTa: entail/neutral/contra),
# so map each logit index onto the fixed [contradicted, unknown, supported] order above.
_id2label = {i: l.lower() for i, l in model.config.id2label.items()}
_label_index = [
    next(i for i, l in _id2label.items() if l.startswith(prefix))
    for prefix in ("contradiction", "neutral", "entailment")
]


def verify_claim(claim, evidence, domain="general", threshold=None):

    with _inference_lock:
        inputs = tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits[:, _label_index],
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

    # Optional heuristic threshold override (e.g. for standalone scripts);
    # by default None so Split Conformal Prediction handles uncertainty mathematically.
    if threshold is not None and confidence < threshold:
        label = "unknown"

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_list
    }


def verify_claims_batch(claim, evidence_texts, domain="general", threshold=None):
    """
    Verifies multiple evidence snippets against a single claim.

    Snippets are scored one forward pass each rather than as one padded batch: retrieved snippets
    vary widely in length (e.g. 64 vs 342 tokens), so padding to the longest wastes most of the
    compute. On Apple M1/MPS this is ~20% faster with bit-identical probabilities (each pair's
    attention mask already excluded padding, so the per-snippet result is unchanged).
    """
    if not evidence_texts:
        return []

    results = []
    with _inference_lock:
        for text in evidence_texts:
            inputs = tokenizer(
                text,
                claim,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits[:, _label_index],
                dim=1
            )[0]

            pred = torch.argmax(probabilities).item()
            conf = probabilities[pred].item()
            prob_list = [probabilities[c].item() for c in range(len(labels))]
            lbl = labels[pred]
            if threshold is not None and conf < threshold:
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
