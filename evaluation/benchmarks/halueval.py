# pyrefly: ignore [missing-import]
from datasets import load_dataset

def load_halueval(split="data"):
    print("Downloading/Loading HaluEval (QA subset) from HuggingFace...")
    dataset = load_dataset(
        "pminervini/HaluEval",
        "qa",
        split=split
    )

    formatted_data = []
    
    # HaluEval QA gives us a 'right_answer' and a 'hallucinated_answer' for each 'question'.
    # We will expand this into our binary SUPPORTS/REFUTES format so our pipeline can blindly test it.
    for row in dataset:
        formatted_data.append({
            "claim": f"Question: {row['question']} Answer: {row['right_answer']}",
            "label": "SUPPORTS"
        })
        formatted_data.append({
            "claim": f"Question: {row['question']} Answer: {row['hallucinated_answer']}",
            "label": "REFUTES"
        })
        
    return formatted_data