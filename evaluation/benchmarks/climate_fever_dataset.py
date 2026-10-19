from datasets import load_dataset


def load_climate_fever(split="test"):
    """Loads the Climate-FEVER dataset (NeurIPS 2020) from Hugging Face.

    Label mapping:
      0: SUPPORTS
      1: REFUTES
      2: NOT ENOUGH INFO
      3: DISPUTED
    Returns list of standardized dict items:
      {'claim': str, 'label': str, 'claim_label': int, 'evidences': list}
    """
    ds = load_dataset("tdiggelm/climate_fever", split=split)
    label_map = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO",
        3: "DISPUTED",
    }
    processed = []
    for item in ds:
        lbl = label_map.get(item.get("claim_label"), "NOT ENOUGH INFO")
        processed.append({
            "claim": item["claim"],
            "label": lbl,
            "claim_label": item.get("claim_label"),
            "evidences": item.get("evidences", []),
        })
    return processed
