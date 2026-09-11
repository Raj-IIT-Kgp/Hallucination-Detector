from datasets import load_dataset



def load_fever(split="validation"):

    dataset = load_dataset(
        "copenlu/fever_gold_evidence",
        split=split
    )

    def extract_evidence(example):
        # Extract the actual evidence text from the nested list structure
        # Example format: [['Article_Name', 'Line_Number', 'Sentence Text']]
        evidence_texts = [ev[2] for ev in example['evidence'] if len(ev) >= 3]
        example['evidence_text'] = " ".join(evidence_texts)
        return example

    dataset = dataset.map(extract_evidence)
    return dataset