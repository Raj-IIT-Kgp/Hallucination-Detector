import json
from retrieval.vector_store import VectorStore
from detector.hallucination_detector import HallucinationDetector


documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Eiffel Tower was completed in 1889.",
    "Gustave Eiffel designed the Eiffel Tower."
]

print("Initializing VectorStore and embedding documents...")
store = VectorStore()
store.add_documents(documents)

detector = HallucinationDetector(store)

answer = """
The Eiffel Tower was completed in 1889.
It is located in London.
It was designed by Gustave Eiffel.
"""

print("Running hallucination detector analysis...")
report = detector.analyze(answer)

print("\n--- FINAL REPORT ---\n")
print(json.dumps(report, indent=4))